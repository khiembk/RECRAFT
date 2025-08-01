import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
from types import SimpleNamespace
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, set_trainable, set_grad_state
from newEmbedder import get_src_train_dataset_1Dmodel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from newEmbedder import wrapper2D, wrapper1D, Embeddings1D, Embeddings2D



class CustomRoberta(torch.nn.Module):
    def __init__(self, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight
        self.output_shape = output_shape

        if isinstance(output_shape, tuple):
            self.dense = True
            print("set dense: ", self.dense)

        
        
        modelname = 'roberta-base' 
        self.model = AutoModel.from_pretrained(modelname)
        self.embedder = nn.Identity()

        
        self.model.embeddings = embedder_placeholder()

        
        self.model.pooler = adaptive_pooler()
        self.predictor = nn.Linear(in_features=768, out_features=output_shape)   
        

        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
            
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, True)


    def forward(self, x):
        
        if self.output_raw:
            return self.embedder(x) 

        # x = self.embedder(x)  
        # # if x.dim() == 1:  # If shape is (hidden_size,)
        # #    x = x.unsqueeze(0).unsqueeze(0)
             
        x = self.model(inputs_embeds=x)['pooler_output']
        x = self.predictor(x)

        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        return x

def get_src_predictor1D(args,root):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    src_num_classes = 9
    #### get source train dataset 
    print("load src model...")
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    src_model = CustomRoberta(src_num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
    #################### prepare for train
    src_model.to(args.device)
    src_model.train() 
    src_model.output_raw = False
    for name, param in src_model.named_parameters():
       if param.requires_grad:
          print(name)
          
    
    num_epochs = 30
    optimizer = optim.AdamW(
         src_model.parameters(),
         lr=args.lr if hasattr(args, 'lr') else 1e-4,
         weight_decay=0.05
     )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
         optimizer,
         T_max= num_epochs
     )
    criterion = nn.CrossEntropyLoss(
         label_smoothing=0.1 if hasattr(args, 'label_smoothing') else 0.0  # Optional smoothing
     )
    for epoch in range(num_epochs):
        running_loss = 0.0 
        
        for i, data in enumerate(src_train_loader):
             x_, y_ = data 
             x_ = x_.to(args.device)
             y_ = y_.to(args.device)
             y_ = y_.long()
             optimizer.zero_grad()
             out = src_model(x_)
             out = F.softmax(out, dim=-1)
             loss = criterion(out, y_)
             loss.backward()
             optimizer.step() 
             running_loss += loss.item()
               # Get the index of max log-probability
             

         
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
               f'Average Loss: {running_loss/len(src_train_loader):.4f}')  

    ######### clean model and return 
    src_predictor = src_model.predictor
    del src_model, src_train_loader
    return src_predictor

############################################################################################################################
