import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from otdd.pytorch.distance import DatasetDistance, FeatureCost
import torch.optim as optim
from math import log
from task_configs import get_data, get_optimizer_scheduler
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
import copy, tqdm
import psutil
from utils import count_params, count_trainable_params, calculate_stats
from torch.utils.data import Subset, DataLoader

def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset))
    return d


class wrapper2D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True
        self.forward_with_feature = False
        print("cur_classification: ", self.classification)
        
        arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
        embed_dim = 128
        output_dim = 1024
        img_size = 224
        patch_size = 4

        if self.classification:
            modelclass = SwinForImageClassification
        else:
            modelclass = SwinForMaskedImageModeling
            
        self.model = modelclass.from_pretrained(arch_name)
        self.model.config.image_size = img_size
        if drop_out is not None:
            self.model.config.hidden_dropout_prob = drop_out 
            self.model.config.attention_probs_dropout_prob = drop_out

        self.model = modelclass.from_pretrained(arch_name, config=self.model.config) if not from_scratch else modelclass(self.model.config)

        if self.classification:
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity()
            self.predictor = nn.Linear(in_features=output_dim, out_features=output_shape)
        else:
            self.pool_seq_dim = adaptive_pooler(output_shape[1] if isinstance(output_shape, tuple) else 1)
            self.pool = nn.AdaptiveAvgPool2d(input_shape[-2:])
            self.predictor = nn.Sequential(self.pool_seq_dim, self.pool)

        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  


    def forward(self, x):
        if self.output_raw:
            return self.model.swin.embeddings(x)[0]

        if self.forward_with_feature:
            return self.forward_with_features(x)

        x = self.model(x).logits
        return self.predictor(x)

    def forward_with_features(self, x):
        z, input_dimensions = self.model.swin.embeddings(x)  
        z = z.clone().requires_grad_(True)
        encoder_output = self.model.swin.encoder(z, input_dimensions)
        #print(f"encoder_output shape: {encoder_output[0].shape}")
        x = self.model.swin.layernorm(encoder_output[0])
        #print(f"layernorm output shape: {x.shape}")
        # Transpose for pooling
        x = x.transpose(1, 2)  # Shape: [batch_size, embed_dim, seq_len]
        #print(f"transposed shape: {x.shape}")
        # Apply pooler
        x = self.model.pooler(x)  # Shape: [batch_size, embed_dim, 1]
        #print(f"pooler output shape: {x.shape}")
        
        # Squeeze for predictor
        x = x.squeeze(-1)  # Shape: [batch_size, embed_dim]
        #print(f"squeezed shape: {x.shape}")
        out = self.predictor(x)
        return out, z


class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.forward_with_feature = False
        self.weight = weight
        self.output_shape = output_shape

        if isinstance(output_shape, tuple):
            self.dense = True

        
        
        modelname = 'roberta-base' 
        configuration = AutoConfig.from_pretrained(modelname)
        if drop_out is not None:
            configuration.hidden_dropout_prob = drop_out
            configuration.attention_probs_dropout_prob = drop_out
        self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)

        if use_embedder:
            self.embedder = Embeddings1D(input_shape, config=self.model.config, embed_dim= 768, target_seq_len= target_seq_len, dense=self.dense)
            embedder_init(self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)    
        else:
            self.embedder = nn.Identity()

        
        self.model.embeddings = embedder_placeholder()
        if self.dense:
            self.model.pooler = nn.Identity()
            self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num, output_shape=output_shape, dense=True)
        else:
            self.model.pooler = adaptive_pooler()
            self.predictor = nn.Linear(in_features=768, out_features=output_shape)   
        

        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
            
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)


    def forward(self, x):
        
        if self.output_raw:
            return self.embedder(x) 

        if self.forward_with_feature:
            return self.forward_with_features(x)

        x = self.embedder(x)

        if self.dense:
            x = self.model(inputs_embeds=x)['last_hidden_state']
            x = self.predictor(x)
        else:
            x = self.model(inputs_embeds=x)['pooler_output']
            x = self.predictor(x)

        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        return x

    def forward_with_features(self, x):
        z = self.embedder(x)
        out = self.embedder(x)
        if self.dense:
            out = self.model(inputs_embeds=out)['last_hidden_state']
            out = self.predictor(x)
        else:
            out = self.model(inputs_embeds=out)['pooler_output']
            out = self.predictor(out)

        if out.shape[1] == 1 and len(out.shape) == 2:
            out = out.squeeze(1)
    

        return out, z


#######################################################################################################
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        cached = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"Allocated GPU memory: {allocated:.2f} MB")
        print(f"Cached GPU memory: {cached:.2f} MB")
    else:
        print("CUDA is not available.")
#######################################################################################################
class Embeddings2D(nn.Module):

    def __init__(self, input_shape, patch_size=4, embed_dim=96, img_size=224, config=None):
        super().__init__()

        self.resize, self.input_dimensions = transforms.Resize((img_size, img_size)), (img_size, img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patched_dimensions = (self.input_dimensions[0] // self.patch_size[0], self.input_dimensions[1] // self.patch_size[1])
        ks = self.patch_size
        self.projection = nn.Conv2d(input_shape[1], embed_dim, kernel_size=ks, stride=self.patch_size, padding=(ks[0]-self.patch_size[0]) // 2)
        self.norm = nn.LayerNorm(embed_dim)
        num_patches = (self.input_dimensions[1] // self.patch_size[1]) * (self.input_dimensions[0] // self.patch_size[0])
        
        conv_init(self.projection)

        
    def maybe_pad(self, x, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            x = nn.functional.pad(x, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            x = nn.functional.pad(x, pad_values)
        return x


    def forward(self, x, *args, **kwargs):
        x = self.resize(x)
        _, _, height, width = x.shape

        x = self.maybe_pad(x, height, width)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)   
        
        return x, self.patched_dimensions


class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, dense=False):
        super().__init__()
        self.dense = dense
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)

        self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
        conv_init(self.projection)


    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim == 768:
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i


    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is None:
            x = inputs_embeds
        b, c, l = x.shape

        x = self.projection(x).transpose(1, 2)
        x = self.norm(x)
            
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps

        if self.embed_dim == 768:
            return x
        else:
            return x, self.patched_dimensions


#########################################################################################################################################################################
def get_pretrain_model2D_feature(args,root,sample_shape, num_classes, source_classes = 10):
    ###################################### train predictor 
    """
    get train model and feature: 
       + get trained predictor but embedder is convolution 
    """
    print("get src_model and src_feature...")
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
    num_classes = 10
    print("num class: ", num_classes)    
    src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
    src_model = src_model.to(args.device)
    src_model.output_raw = False
    optimizer = optim.AdamW(
         src_model.parameters(),
         lr=args.lr if hasattr(args, 'lr') else 1e-4,
         weight_decay=0.05
     )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
         optimizer,
         T_max=args.embedder_epochs/10
     )
    criterion = nn.CrossEntropyLoss(
         label_smoothing=0.1 if hasattr(args, 'label_smoothing') else 0.0  # Optional smoothing
     )
    src_model.train()
    set_grad_state(src_model.predictor, True)
    print("trainabel params count :  ",count_trainable_params(src_model))
    print("trainable params: ")  
    for name, param in src_model.named_parameters():
       if param.requires_grad:
          print(name)
     #print("model architechture: ", src_model)      
    for epoch in range(args.embedder_epochs//10):
        running_loss = 0.0 
        correct = 0  
        total = 0
        for i, data in enumerate(src_train_loader):
             x_, y_ = data 
             x_ = x_.to(args.device)
             y_ = y_.to(args.device)
             x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
             optimizer.zero_grad()
             out = src_model(x_)
             loss = criterion(out, y_)
             loss.backward()
             optimizer.step()
             running_loss += loss.item()
             _, predicted = torch.max(out, 1)  # Get the index of max log-probability
             total += y_.size(0)
             correct += (predicted == y_).sum().item()
         
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{args.embedder_epochs//10}], '
               f'Average Loss: {running_loss/len(src_train_loader):.4f}'
               f' Accuracy: {accuracy:.2f}%')  
         
    ##### set output_raw 
    src_model.output_raw = True
    src_model.eval()
    ##### get source feature from cifar10
    src_feats = []
    src_ys = []
    for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)

            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
            
    src_feats = torch.cat(src_feats, 0)
    src_ys = torch.zeros(len(src_feats), dtype=torch.long)
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    ##### clearn cache
    del src_ys, src_feats, src_train_loader
    torch.cuda.empty_cache()
    
    return src_model, src_train_dataset


###############################################################################################################################################
def get_pretrain_model2D_feature_with_tau(args, root, sample_shape, num_classes, source_classes=10, tau=0.4, rho=100):
    ###################################### train predictor 
    """
    Retrain the source prediction head with tau to enforce the assumption
    Regularizer: L = L_s + (rho / N_s) * sum(max(0, ||grad_z l_s(z)||_2 - tau)^2)
    where z is the output of the embedder
    """
    print("get src_model and src_feature...")
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
    num_classes = 10
    print("num class: ", num_classes)    
    src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
    src_model = src_model.to(args.device)
    src_model.output_raw = False
    
    
    
    
    
    optimizer = optim.AdamW(
        src_model.parameters(),
        lr=args.lr if hasattr(args, 'lr') else 1e-4,
        weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max= 20
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1 if hasattr(args, 'label_smoothing') else 0.0
    )
    src_model.train()
    set_grad_state(src_model.predictor, True)
    print("trainable params count: ", count_trainable_params(src_model))
    print("trainable params: ")  
    for name, param in src_model.named_parameters():
        if param.requires_grad:
            print(name)
    ############################################################################################
    for epoch in range(6):
        running_loss = 0.0 
        correct = 0  
        total = 0
        for i, data in enumerate(src_train_loader):
             x_, y_ = data 
             x_ = x_.to(args.device)
             y_ = y_.to(args.device)
             x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
             optimizer.zero_grad()
             out = src_model(x_)
             loss = criterion(out, y_)
             loss.backward()
             optimizer.step()
             running_loss += loss.item()
             _, predicted = torch.max(out, 1)  # Get the index of max log-probability
             total += y_.size(0)
             correct += (predicted == y_).sum().item()
         
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{args.embedder_epochs//10}], '
               f'Average Loss: {running_loss/len(src_train_loader):.4f}'
               f' Accuracy: {accuracy:.2f}%')  
    #########################################################################################################
    #### Enable foward with feature
    src_model.forward_with_feature = True
    ################################

    for epoch in range(6,20):
        running_loss = 0.0 
        running_reg_loss = 0.0  # Track regularizer loss
        correct = 0  
        total = 0
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            y_ = y_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            
            optimizer.zero_grad()
            out, z = src_model(x_)  # Get both output and embedder features
            z.requires_grad_(True)  
            # Compute source loss (cross-entropy)
            loss_s = criterion(out, y_)
            
            # Compute gradient of loss w.r.t. features z
            grad_z = torch.autograd.grad(loss_s, z, create_graph=True, allow_unused=False)[0]
            
            # Compute L2 norm of gradient for each sample in the batch
            grad_norm = torch.norm(grad_z, p=2, dim=1)  # Shape: [batch_size]
            
            # Compute regularizer: sum(max(0, ||grad_z||_2 - tau)^2)
            reg_term = torch.max(torch.zeros_like(grad_norm), grad_norm - tau) ** 2
            reg_loss = (rho / y_.size(0)) * reg_term.sum()  # Scale by rho / N_s
            
            # Total loss: L = L_s + reg_loss
            total_loss = loss_s + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += loss_s.item()
            running_reg_loss += reg_loss.item()
            _, predicted = torch.max(out, 1)  # Get the index of max log-probability
            total += y_.size(0)
            correct += (predicted == y_).sum().item()
        
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{args.embedder_epochs}], '
              f'Average Source Loss: {running_loss/len(src_train_loader):.4f}, '
              f'Average Reg Loss: {running_reg_loss/len(src_train_loader):.4f}, '
              f'Accuracy: {accuracy:.2f}%')  

    return src_model


##############################################################################################################################################
def feature_matching_OTDD_tgt_model(args,root , tgt_model, sample_shape, num_classes):
    """
    Embedder training minimize source and target feture distribution:  
      + return tgt_model
    """ 
    
    print("feature matching by OTDD...")
    ##### check tgt_model
    set_grad_state(tgt_model.embedder, True)
    print("trainabel params count :  ",count_trainable_params(tgt_model))
    print("trainable params: ")  
    for name, param in tgt_model.named_parameters():
      if param.requires_grad:
         print(name)
    ### load tgt_loader
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    if len(sample_shape) == 4:
        IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
            
        src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
        src_model = src_model.to(args.device).eval()
            
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)

            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
        src_feats = torch.cat(src_feats, 0)
        src_ys = torch.cat(src_ys, 0).long()
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
        del src_model    

    else:
        src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 

    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    
    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()

    
    score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    
    score = 0
    total_losses, times, embedder_stats = [], [], []
    
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()

        for i in np.random.permutation(num_classes_new):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to(args.device)
                out = tgt_model(x)
                feats.append(out)
                datanum += x.shape[0]
                
                if datanum > args.maxsamples: break

            feats = torch.cat(feats, 0).mean(1)
            if feats.shape[0] > 1:
                loss = tgt_class_weights[i] * score_func(feats)
                loss.backward()
                total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader, tgt_train_loaders
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model
##############################################################################################################################################
def feature_matching_tgt_model(args,root , tgt_model, src_train_dataset):
    """
    Embedder training minimize source and target feture distribution:  
      + return tgt_model
    """ 
    
    print("feature matching....")
    ##### check tgt_model
    set_grad_state(tgt_model.embedder, True)
    print("trainabel params count :  ",count_trainable_params(tgt_model))
    print("trainable params: ")  
    for name, param in tgt_model.named_parameters():
      if param.requires_grad:
         print(name)
    ### load tgt_loader
    print("load tgt dataset...") 
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    #### fix fowrad flow and set trainable to tgt_model
    tgt_model.output_raw = True
    tgt_model = tgt_model.to(args.device).train()
    
    #### init score function 
    score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    ##### Test for new test function
    # src_feature =  src_train_dataset.tensors[0].numpy()
    # new_score_func = partial(OptimalTV, sample1 = src_feature)
    ##### get optimizer
    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()
    ##### train embedder with score_func
     
    total_losses, times, embedder_stats = [], [], []
    print("begin feature matching...")
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()
        feats = []
        datanum = 0
        ##### shuffle dataset 
        
        shuffled_loader = torch.utils.data.DataLoader(
                tgt_train_loader.dataset,
                batch_size=tgt_train_loader.batch_size,
                shuffle=True,  # Enable shuffling to permute the order
                num_workers=tgt_train_loader.num_workers,
                pin_memory=tgt_train_loader.pin_memory)
        ##### begin training
             
        for j, data in enumerate(shuffled_loader):
            
            if transform is not None:
                x, y, z = data
            else:
                x, y = data 
                
            x = x.to(args.device)
            out = tgt_model(x)
            
            feats.append(out)
            datanum += x.shape[0]
                
            if datanum > args.maxsamples:
                  break
        
        feats = torch.cat(feats, 0).mean(1)
        if feats.shape[0] > 1:
            #feats_np = feats.numpy()
            loss = (len(feats)/len(tgt_train_loader))*score_func(feats)
            loss.backward()
            total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tWasserstein distance  loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader 
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model
###############################################################################################################################################    
def compute_entropy_over_dataset(args,tgt_train_loader, src_model, transform = None, max_size = 1000):
    """ Compute entropy over dataset without gradient computing.

    Returns:
        H(Y^s): float
    """
    #### shffed_loader
    shuffled_loader = torch.utils.data.DataLoader(
                tgt_train_loader.dataset,
                batch_size=tgt_train_loader.batch_size,
                shuffle=True,  # Enable shuffling to permute the order
                num_workers=tgt_train_loader.num_workers,
                pin_memory=tgt_train_loader.pin_memory)
    
    native_prob = []
    native_num = 0
    max_size = min(max_size,len(shuffled_loader))
    for j, data in enumerate(shuffled_loader):
                
            if transform is not None:
                  x, y, z = data
            else:
                  x, y = data 
            
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            with torch.no_grad():
                out = src_model(x)
                out = F.softmax(out, dim=-1)
                native_prob.append(out)
            native_num += x.shape[0]     
            if native_num >= max_size:
                #    print("run backward: ")
                #    get_gpu_memory_usage()
                native_probs_tensor = torch.cat(native_prob, dim=0)
                loss = - Entropy_loss(native_probs_tensor)
                del native_probs_tensor, native_prob
                break
    
    del shuffled_loader
    return loss          
#################################################################################################################################################    
def Entropy_loss(dummy_probs, epxilon = 1e-10):
    """
    Return entropy loss given dummy label and dummpy prob.
    Args:
        dummy_probs: Tensor [batch_size], probabilities for dummy_labels (e.g., [0.9, 0.8, 0.7])
       
    Return H(Y^s)
    """
    dummy_probs = dummy_probs.to(dtype=torch.float32)
    # 
    # Calculate mean probabilities across batch dimension (dim=0)
    P = torch.mean(dummy_probs, dim=0)
    
    # Calculate entropy: -∑(p * log(p))
    # Adding small epsilon (1e-10) to avoid log(0)
    entropy = -torch.sum(P * torch.log(P + epxilon))
    
    return entropy
###################################################################################################################################################
def label_matching_by_entropy(args,root, src_model, tgt_embedder,num_classes ,src_num_classes= 10, model_type = "2D", fronzen_predictor = True):
    """ Label matching by minimize src entrpy each classes sum H(Y^S|Y^T = y^t)

    Args:
        args (_type_): _description_
        root (_type_): path to dataset 
        src_model (_type_): _description_
        tgt_embedder (_type_): _description_
        num_classes (_type_): _description_
        src_num_classes (_type_): _description_
        model_type: 1D or 2D backbone
    Returns:
        loss_value : sum H(Y^S|Y^T = y^t)
    """
    print("[Entropy] label matching with src model...")
    ##### check src_model set trainable params
    if model_type == "2D":
       src_model.embedder = tgt_embedder
       src_model.model.swin.embeddings = src_model.embedder  
    else:
       src_model.embedder = tgt_embedder 
         
    set_grad_state(src_model.model,False)
    set_grad_state(src_model.embedder, True)
    if fronzen_predictor:
       print("[Label Matching]-Set src predictor frozen...")
       set_grad_state(src_model.predictor, False)
    ################################################################    
    print("trainabel params count :  ",count_trainable_params(src_model))
    print("trainable params: ")
    src_model.output_raw = False
    src_model = src_model.to(args.device).train()  
    # for name, param in src_model.named_parameters():
    #   if param.requires_grad:
    #      print(name)
         
    # ##### load tgt dataset
    # print(src_model)
    print("load tgt dataset...")
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    

    if args.dataset == "PSICOV":
       print("Sample sub datset for Psicov...")
       subset_dataset = torch.utils.data.Subset(tgt_train_loader.dataset, range(args.lb_samples))
       tgt_train_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1)
       del subset_dataset 

    print("infer label...")
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader, dataset= args.dataset )
        
    else: 
        num_classes_new = num_classes
        
    print("load target dataset by classes...")
    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)
    
    ####### get optimizer
    args, src_model, optimizer, scheduler = get_optimizer_scheduler(args, src_model, module=None, n_train=n_train)
    optimizer.zero_grad()         
    ####### train with dummy label 
    print("Training with dummy label...")
    ###### config for testing
    label_matching_ep = args.label_epochs
    max_sample = args.label_maxsamples
    total_losses, times, stats = [], [], []
    
    
    for ep in range(label_matching_ep):
        total_loss = 0    
        time_start = default_timer()    
        
        for i in np.random.permutation(num_classes_new):
            
            datanum = 0
            dummy_probability = []
            for j, data in enumerate(tgt_train_loaders[i]):
                
               if transform is not None:
                  x, y, z = data
               else:
                  x, y = data 
                
               x = x.to(args.device)
               y = y.to(args.device)
               out = src_model(x)
               out = F.softmax(out, dim=-1)
               dummy_probability.append(out)
               datanum += x.shape[0]
            #    print("datanum: ", datanum)
            #    get_gpu_memory_usage() 
               if datanum >= max_sample:
                #    print("run backward: ")
                #    get_gpu_memory_usage()
                   dummy_probs_tensor = torch.cat(dummy_probability, dim=0)
                   loss = tgt_class_weights[i]*(datanum/len(tgt_train_loaders[i]))*Entropy_loss(dummy_probs_tensor)
                   loss.backward()
                   optimizer.step()
                   optimizer.zero_grad()
                   total_loss += loss.item()
                   dummy_probability = []
                   datanum = 0
                   #print("after grad")
                   #get_gpu_memory_usage()
        ###################### handle leftover dataset. 
            if (datanum >= max_sample//2):             
                dummy_probs_tensor = torch.cat(dummy_probability, dim=0)  # This is a tensor
                loss = tgt_class_weights[i]*(datanum/len(tgt_train_loaders[i]))*Entropy_loss(dummy_probs_tensor)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
        ##############################
        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        stats.append([total_losses[-1], times[-1]])
        print("[label matching ", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tCE loss:", "%.4f" % total_losses[-1])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()           

    ### delete trash
    del tgt_train_loader,tgt_train_loaders        
    return src_model
##################################################################################################################################################
def create_native_set(tgt_train_loader,i, desired_negative_size):
    
    shuffled_loader = torch.utils.data.DataLoader(
                tgt_train_loader.dataset,
                batch_size=tgt_train_loader.batch_size,
                shuffle=True,  
                num_workers=tgt_train_loader.num_workers,
                pin_memory=tgt_train_loader.pin_memory)
    
    negative_indices = []
    negative_samples_collected = 0 
    
    for neg_batch in shuffled_loader:
        neg_inputs, neg_labels = neg_batch  # Adjust based on your data structure
        neg_mask = neg_labels != i
        neg_indices_batch = torch.where(neg_mask)[0]
        negative_indices.extend(neg_indices_batch.tolist())
        negative_samples_collected += neg_indices_batch.size(0)
        
        if negative_samples_collected >= desired_negative_size:
            break
    
    negative_dataset = Subset(shuffled_loader.dataset, negative_indices[:desired_negative_size])
    cur_negative_set = DataLoader(
            negative_dataset,
            batch_size= tgt_train_loader.batch_size,
            shuffle=True,  # Shuffle the negative set
            num_workers=tgt_train_loader.num_workers,
            pin_memory=tgt_train_loader.pin_memory
        )
    return cur_negative_set      
#########################################################################################################################################################################
def create_uniform_native_set(tgt_train_loader, exclude_class, desired_negative_size):
    # Get the full dataset and its labels
    dataset = tgt_train_loader.dataset
    all_labels = None
    
    # Assuming dataset yields (inputs, labels) or (inputs, labels, extras)
    # Load all labels efficiently (assuming they're accessible)
    for batch in DataLoader(dataset, batch_size= tgt_train_loader.batch_size, shuffle=False):
        if len(batch) == 3:  # With transform
            _, labels, _ = batch
        else:  # Without transform
            _, labels = batch
        if all_labels is None:
            all_labels = labels
        else:
            all_labels = torch.cat([all_labels, labels], dim=0)

    # Number of classes excluding the current one
    num_classes = len(torch.unique(all_labels))  # Assumes labels are 0 to num_classes-1
    num_classes_excluding_i = num_classes - 1
    samples_per_class = desired_negative_size // num_classes_excluding_i  # Uniform split

    # Collect indices for each class
    negative_indices = []
    for class_idx in range(num_classes):
        if class_idx == exclude_class:
            continue  # Skip the excluded class

        # Find indices for this class
        class_mask = all_labels == class_idx
        class_indices = torch.where(class_mask)[0].tolist()
        num_available = len(class_indices)

        # Sample from this class
        if num_available <= samples_per_class:
            # Take all if fewer than needed, pad with random repeats if necessary
            selected_indices = class_indices
            if num_available < samples_per_class:
                extra_indices = np.random.choice(class_indices, size=samples_per_class - num_available, replace=True).tolist()
                selected_indices.extend(extra_indices)
        else:
            # Randomly sample without replacement
            selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False).tolist()

        negative_indices.extend(selected_indices)

    # Adjust total size to match desired_negative_size
    current_size = len(negative_indices)
    if current_size < desired_negative_size:
        # Add random samples from non-excluded classes to fill
        all_non_i_indices = torch.where(all_labels != exclude_class)[0].tolist()
        extra_indices = np.random.choice(all_non_i_indices, size=desired_negative_size - current_size, replace=True).tolist()
        negative_indices.extend(extra_indices)
    elif current_size > desired_negative_size:
        # Trim excess randomly
        negative_indices = np.random.choice(negative_indices, size=desired_negative_size, replace=False).tolist()

    # Create the negative dataset
    negative_dataset = Subset(dataset, negative_indices)
    cur_negative_set = DataLoader(
        negative_dataset,
        batch_size=tgt_train_loader.batch_size,
        shuffle=True,
        num_workers=tgt_train_loader.num_workers,
        pin_memory=tgt_train_loader.pin_memory
    )
    return cur_negative_set   
#########################################################################################################################################################################
def label_matching_by_conditional_entropy(args,root, src_model, tgt_embedder,num_classes ,src_num_classes= 10, model_type = "2D", frozen_predictor = True):

    """ Label matching by minimize src conditional entrpy each classes (sum H(Y^S|Y^T = y^t)) - H(Y^s)

    Args:
        args (_type_): _description_
        root (_type_): path to dataset 
        src_model (_type_): _description_
        tgt_embedder (_type_): _description_
        num_classes (_type_): _description_
        src_num_classes (_type_): _description_
        model_type: 1D or 2D backbone
    Returns:
        loss_value : (sum H(Y^S|Y^T = y^t)) - H(Y^s)
    """
    print("[Conditional Entropy] label matching with src model...")
    ##### check src_model set trainable params
    if model_type == "2D":
       src_model.embedder = tgt_embedder
       src_model.model.swin.embeddings = src_model.embedder 
    else:
       src_model.embedder = tgt_embedder 
          
    set_grad_state(src_model.model, False)
    set_grad_state(src_model.embedder, True)
    if frozen_predictor:
       print("[Label Matching]-Set src predictor frozen")
       set_grad_state(src_model.predictor, False) 
    print("trainabel params count :  ",count_trainable_params(src_model))
    print("trainable params: ")
    src_model.output_raw = False
    src_model = src_model.to(args.device).train()  
    # for name, param in src_model.named_parameters():
    #   if param.requires_grad:
    #      print(name)
         
    ##### load tgt dataset
    # print(src_model)
    print("load tgt dataset...")
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    if args.dataset == "PSICOV":
        print("Sample random subset of dataset for PSICOV...")
        subset_dataset = torch.utils.data.Subset(tgt_train_loader.dataset, range(args.lb_samples))
        tgt_train_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1)
        del subset_dataset  

    print("infer label...")
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader, dataset= args.dataset)
        
    else: 
        num_classes_new = num_classes
        
    print("load target dataset by classes...")
    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)
    
    ####### get optimizer
    args, src_model, optimizer, scheduler = get_optimizer_scheduler(args, src_model, module=None, n_train=n_train)
    optimizer.zero_grad()         
    ####### train with dummy label 
    print("Training with dummy label...")
    ###### config for testing
    label_matching_ep = args.label_epochs
    max_sample = args.label_maxsamples
    neg_losses,pos_losses, times, stats = [], [], [] ,[]
    ####################################
    #init suffled loader
    
    
    for ep in range(label_matching_ep):
        pos_loss = 0  
        neg_loss = 0  
        time_start = default_timer()    
        
        for i in np.random.permutation(num_classes_new):
            #print("curent class: ",i)
            ##############################################
            datanum = 0
            dummy_probability = []
            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                  x, y, z = data
                else:
                  x, y = data 
                
                #### load sample to gpu
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
               #### No gradients during inference
               
                out = src_model(x)
                out = F.softmax(out, dim=-1)
                dummy_probability.append(out)
                datanum += x.shape[0]
            #    print("datanum: ", datanum)
            #    get_gpu_memory_usage() 
                if datanum >= max_sample:
                #    print("run backward on positive: ")
                #    get_gpu_memory_usage()
                   dummy_probs_tensor = torch.cat(dummy_probability, dim=0)
                   loss = tgt_class_weights[i]*(datanum/len(tgt_train_loaders[i]))*Entropy_loss(dummy_probs_tensor)
                   loss.backward()
                   optimizer.step()
                   optimizer.zero_grad()
                   pos_loss += loss.item()
                   # Clear memory
                   del dummy_probs_tensor, dummy_probability
                   dummy_probability = []
                   datanum = 0
                   torch.cuda.empty_cache()
                #    print("after pos grad")
                #    get_gpu_memory_usage()
        ###################### handle leftover dataset. 
            if (datanum >= max_sample//2):             
                dummy_probs_tensor = torch.cat(dummy_probability, dim=0)  # This is a tensor
                loss = tgt_class_weights[i]*(datanum/len(tgt_train_loaders[i]))*Entropy_loss(dummy_probs_tensor)
                loss.backward()
                pos_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                del dummy_probs_tensor, dummy_probability
        ##############################
            ###### code for nativate sample 
            desired_negative_size = tgt_train_loaders[i].batch_size * len(tgt_train_loaders[i]) 
            native_dataset = create_native_set(tgt_train_loader,i,desired_negative_size)
            #### train with native set
            neg_num = 0
            neg_prob = []
            neg_loss = 0
            for k, neg_data in enumerate(native_dataset):
                if transform is not None:
                    x_neg, y_neg, z_neg = neg_data
                else:
                    x_neg, y_neg = neg_data 
                
                #### load sample to gpu
                x_neg = x_neg.to(args.device, non_blocking=True)
                y_neg = y_neg.to(args.device, non_blocking=True)
               #### No gradients during inference
               
                out_neg = src_model(x_neg)
                out_neg = F.softmax(out_neg, dim=-1)
                neg_prob.append(out_neg)
                neg_num += x_neg.shape[0]
                if neg_num >= max_sample:
                #    print("neg_num = ", neg_num)
                #    print("run backward on neg: ")
                #    get_gpu_memory_usage()
                   neg_prob_tensor = torch.cat(neg_prob, dim=0)
                   loss_neg = -tgt_class_weights[i]*(neg_num/len(native_dataset))*Entropy_loss(neg_prob_tensor)
                   loss_neg.backward()
                   optimizer.step()
                   optimizer.zero_grad()
                   neg_loss += loss_neg.item()
                   # Clear memory
                   del neg_prob, neg_prob_tensor
                   neg_prob = []
                   neg_num = 0
                   torch.cuda.empty_cache()
                #    print("after neg grad")
                #    get_gpu_memory_usage()
            del native_dataset          
        #neg_loss = compute_entropy_over_dataset(args,tgt_train_loader,src_model,transform)
        ##############################
        time_end = default_timer()  
        times.append(time_end - time_start) 

        pos_losses.append(pos_loss)
        neg_losses.append(neg_loss)
        stats.append([pos_losses[-1], times[-1]])
        print("[label matching ", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tCE pos loss:", "%.4f" % pos_losses[-1],"\tCE neg loss:", "%.4f" % neg_losses[-1])
        ### delete native set
    
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()           

    ### delete trash
    del tgt_train_loader,tgt_train_loaders        
    return src_model     
#########################################################################################################################################################################
def get_src_train_dataset_1Dmodel(args,root):
    """
    get source train dataset with backbone Roberta: 
       + return source train dataset.
    """
    
    ####### load src train dataset.
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
    zero_labels = torch.zeros_like(src_ys)
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, zero_labels)
    return src_train_dataset

#########################################################################################################################################################################
def get_tgt_model(args, root, sample_shape, num_classes, loss, add_loss=False, use_determined=False, context=None, opid=0):
    
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    if len(sample_shape) == 4:
        IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
            
        src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
        src_model = src_model.to(args.device).eval()
            
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)

            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
        src_feats = torch.cat(src_feats, 0)
        src_ys = torch.cat(src_ys, 0).long()
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
        del src_model    

    else:
        src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 

    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()

    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()

    if args.objective == 'otdd-exact':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    elif args.objective == 'otdd-gaussian':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    elif args.objective == 'l2':
        score_func = partial(l2, src_train_dataset=src_train_dataset)
    else:
        score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    
    score = 0
    total_losses, times, embedder_stats = [], [], []
    
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()

        for i in np.random.permutation(num_classes_new):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to(args.device)
                out = tgt_model(x)
                feats.append(out)
                datanum += x.shape[0]
                
                if datanum > args.maxsamples: break

            feats = torch.cat(feats, 0).mean(1)
            if feats.shape[0] > 1:
                loss = tgt_class_weights[i] * score_func(feats)
                loss.backward()
                total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader, tgt_train_loaders
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model, embedder_stats


def infer_labels(loader, k = 10 , dataset = None):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    if hasattr(loader.dataset, 'tensors'):
        X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
        try:
            Z = loader.dataset.tensors[2].cpu()
        except:
            Z = None
    else:
        if dataset == "PSICOV":
           X, Y, Z = get_tensors_psicov(loader.dataset) 
        else:     
           X, Y, Z = get_tensors(loader.dataset)

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    print("class weights")
    for target, subset in subsets.items():
        print(target, len(subset), len(train_set), len(subset)/len(train_set))

    return loaders, class_weights



def get_tensors(dataset):
    xs, ys, zs = [], [], []
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        xs.append(np.expand_dims(data[0], 0))
        ys.append(np.expand_dims(data[1], 0))
        if len(data) == 3:
            zs.append(np.expand_dims(data[2], 0))

    xs = torch.from_numpy(np.array(xs)).squeeze(1)
    ys = torch.from_numpy(np.array(ys)).squeeze(1)

    if len(zs) > 0:
        zs = torch.from_numpy(np.array(zs)).squeeze(1)
    else:
        zs = None

    return xs, ys, zs

def get_tensors_psicov(dataset, max_samples=430, batch_size=4):
    print(f"Starting get_tensors, max_samples={max_samples}, memory used: {psutil.virtual_memory()}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    xs, ys, zs = [], [], []
    samples_processed = 0
    
    for batch in loader:
        print(f"Processing batch, samples_processed={samples_processed}, memory used: {psutil.virtual_memory()}")
        X_batch, Y_batch = batch[0], batch[1]
        Z_batch = batch[2] if len(batch) > 2 else None
        
        xs.append(X_batch.cpu())
        ys.append(Y_batch.cpu().numpy())
        if Z_batch is not None:
            zs.append(Z_batch.cpu())
            
        samples_processed += X_batch.shape[0]
        if samples_processed >= max_samples:
            break
    
    print(f"Concatenating tensors, memory used: {psutil.virtual_memory()}")
    xs = torch.cat(xs) if xs else torch.tensor([])
    ys = np.concatenate(ys) if ys else np.array([])
    zs = torch.cat(zs) if zs else None
    
    print(f"Final shapes - X: {xs.shape}, Y: {ys.shape}, Z: {zs.shape if zs is not None else None}")
    return xs, ys, zs