import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn # type: ignore
from timeit import default_timer
from tqdm import tqdm
import yaml
from types import SimpleNamespace
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, set_trainable, set_grad_state
from utils import count_params, count_trainable_params, calculate_stats
from newEmbedder import get_pretrain_model2D_feature, wrapper1D, wrapper2D, feature_matching_tgt_model,get_src_train_dataset_1Dmodel
from test_model import get_src_predictor1D
import wandb
from datetime import datetime
from newEmbedder import label_matching_by_entropy, label_matching_by_conditional_entropy


def main(use_determined ,args,info=None, context=None, DatasetRoot= None, log_folder = None, second_train = False):

    ############## Init log file and set seed
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #args.device = 'cuda' 
    print("The current device is: ", args.device)
    root = '/datasets' if use_determined else './datasets'
    if (DatasetRoot != None):
        if (args.pde):
           root = DatasetRoot
        else:    
           root = DatasetRoot + '/datasets'

    print("Path folder dataset: ",root) 
    ############################### Set seed ###############################
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)
    
    ############################################### Log ################################################
    log_file = f"{args.dataset}_{args.finetune_method}.log"
    if (log_folder is not None):
        log_dir = os.path.join(log_folder)
        os.makedirs(log_dir, exist_ok= True)
        log_file = os.path.join(log_dir, f"{args.dataset}_{args.finetune_method}.log")
    else:
        log_folder = "Logs"
        log_dir = os.path.join(log_folder)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{args.dataset}_{args.finetune_method}.log")

    logging.basicConfig(filename= log_file,
                    level=logging.INFO,  # Set logging level
                    format='%(asctime)s - %(levelname)s - %(message)s') 
    ###################################################################################################
    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    ##################################################################################################
    ##### Init wanDB
    wandb.login(key= "87a17a462a0003e50590ec537dda9beacbcc2d63")
    
    wandb.init(
      # Set the project where this run will be logged
      project= f"CrossModality_{args.dataset}",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name = (
    f"experiment_{args.dataset}_id_{str(args.experiment_id)}_"
    f"CE-{args.C_entropy}_LE-{args.label_epochs}_seed-{args.seed}"),
      # Track hyperparameters and run metadata
      config={
      "optimizer": args.optimizer,
      "back_bone": args.weight,
      "target_dataset": args.dataset,
      "training_epochs": args.epochs,
      "feature_matching_epochs:": args.embedder_epochs,
      "Conditional_entropy": args.C_entropy,
      "label_matching_epochs:": args.label_epochs,
      })
    
    #################### Load config 
    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    print("current configs: ", args)
    if load_embedder(use_determined, args):
        print("Log: Set embedder_epochs = 0")
        args.embedder_epochs = 0
    ####### determind type of backbone
    Roberta = True if len(sample_shape) == 3 else False
    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    ########## init tgt_model
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()
    continue_training = check_if_continue_training(args)
    if continue_training:
       print("continue training...")
       print("set model trainable...")
       set_grad_state(tgt_model.model, True)
       tgt_model.output_raw = False
    else:   
       if Roberta is not True: 
        print("2D task...")
    ######### config for testing 
        
        src_num_classes = 10  
    ######### get src_model and src_feature
        src_model, src_train_dataset = get_pretrain_model2D_feature(args,root,sample_shape,num_classes,src_num_classes)
    
    ######### feature matching for tgt_model.
        tgt_model = feature_matching_tgt_model(args,root, tgt_model,src_train_dataset)
        del src_train_dataset
    ######### label matching for src_model.
        if args.C_entropy == False:
            print("Do fast label matching...")
            src_model = label_matching_by_entropy(args,root, src_model, tgt_model.embedder, num_classes, model_type="2D")
        else:
            print("Do conditional label matching...")
            src_model = label_matching_by_conditional_entropy(args,root, src_model, tgt_model.embedder, num_classes, model_type="2D")    
    ######### fine-tune all tgt_model after feature-label matching.
        print("Init tgt_model backbone by src_model...")
        tgt_model.model.swin.encoder = src_model.model.swin.encoder
        del src_model
        set_grad_state(tgt_model.model, True)
        set_grad_state(tgt_model.embedder, True)
        
       else:
        print("1D task...")
        #### get source train dataset 
        src_train_dataset = get_src_train_dataset_1Dmodel(args,root)
        ########### feature matching for target model.
        tgt_model = feature_matching_tgt_model(args,root, tgt_model,src_train_dataset)
        del src_train_dataset
        ########## label matching for 1D task
        print("get src predictor...")
        src_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
        src_model.predictor = get_src_predictor1D(args,root)
        if args.C_entropy == False:
            print("Do fast label matching...")
            src_model = label_matching_by_entropy(args, root, src_model, tgt_model.embedder, num_classes, model_type="1D")
        else:
            print("Do conditional label matching...")
            src_model = label_matching_by_conditional_entropy(args, root, src_model, tgt_model.embedder, num_classes, model_type="1D")    
        ####################################################################################
        #Init tgt body model
        tgt_model.model.encoder = src_model.model.encoder
        del src_model
        set_grad_state(tgt_model.model, True)
        set_grad_state(tgt_model.embedder, True)
        ######################################################
    
         
    print("final model: ", tgt_model)

    ######### load tgt dataset 
    print("load tgt dataset ...")
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    ###############################################################################################
    
    print("load dic if model was trained ...")
    tgt_model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, tgt_model, None, None, n_train, freq=args.validation_freq, test=True)
    # embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    offset = 0 if ep_start == 0 else 1
    args, tgt_model, optimizer, scheduler = get_optimizer_scheduler(args, tgt_model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs
   
    if args.device == 'cuda':
        tgt_model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    print("\n------- Experiment Summary --------")
    print("id:", args.experiment_id)
    print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer["params"]["lr"])
    print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    print("finetune method:", args.finetune_method)
    print("all param count:", count_params(tgt_model),)
    logging.info("all param count: %d", count_params(tgt_model))
    print("trainabel params count: %d  ",count_trainable_params(tgt_model))
    logging.info("trainabel params count:  %d  ",count_trainable_params(tgt_model))
    # print("print model")
    # print(model)
    logging.info(f"Model Structure:\n{tgt_model}")
    ###### load state optimizer
    if second_train:
        tgt_model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, tgt_model, None, None, n_train, freq=args.validation_freq, test = True)
    else:
        tgt_model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, tgt_model, optimizer, scheduler, n_train, freq=args.validation_freq)
    #embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    train_time = []
    embedder_stats = []
    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")

    for ep in range(ep_start, args.epochs + args.predictor_epochs):
        if not train_full and ep >= args.predictor_epochs:
            args, tgt_model, optimizer, scheduler = get_optimizer_scheduler(args, tgt_model, module=None, n_train=n_train)
            train_full = True

        time_start = default_timer()

        train_loss = train_one_epoch(context, args, tgt_model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform)
        train_time_ep = default_timer() -  time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
            val_loss, val_score = evaluate(context, args, tgt_model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))
            logging.info(
            "[train %s %d %.6f] time elapsed: %.4f\ttrain loss: %.4f\tval loss: %.4f\tval score: %.4f\tbest val score: %.4f",
            "full" if ep >= args.predictor_epochs else "predictor",ep,optimizer.param_groups[0]['lr'], train_time[-1], train_loss,
             val_loss, val_score,compare_metrics(train_score))
            wandb.log({
            "epoch": ep,
            "time_elapsed": train_time[-1],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_score": val_score,
            "best_val_score": compare_metrics(train_score)
            })
            if use_determined :
                id_current = save_state(use_determined, args, context, tgt_model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                try:
                    context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                    context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                except:
                    pass
                    
            if compare_metrics(train_score) == val_score:
                if not use_determined :
                    print("save state at epoch ep: ", ep)
                    id_current = save_state(use_determined, args, context, tgt_model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                id_best = id_current
            

        if ep == args.epochs + args.predictor_epochs - 1:
            print("\n------- Start Test --------")
            test_scores = []
            test_model = tgt_model
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            logging.info("[test last]\ttime elapsed: %.4f\ttest loss: %.4f\ttest score: %.4f",test_time_end - test_time_start,test_loss,test_score)

            test_model, _, _, _, _, _ = load_state(use_determined, args, context, test_model, optimizer, scheduler, n_train, id_best, test=True)
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            logging.info("[test best-validated]\ttime elapsed: %.4f\ttest loss: %.4f\ttest score: %.4f" % (test_time_end - test_time_start, test_loss, test_score))
            

            if use_determined:
                checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
                with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    np.save(os.path.join(path, 'test_score.npy'), test_scores)
            else:
                path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
                np.save(os.path.join(path, 'test_score.npy'), test_scores)

           
        if use_determined and context.preempt.should_preempt():
            print("paused")
            return

   
    wandb.finish()    


def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, decoder=None, transform=None):    

    model.train()             
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(tqdm(loader, desc="Training Progress", leave=True)):

        if transform is not None:
            x, y, z = data
            z = z.to(args.device)
        else:
            x, y = data 
        
        x, y = x.to(args.device), y.to(args.device)
        out = model(x)

        if isinstance(out, dict):
            out = out['out']

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)

        if args.dataset[:4] == "DRUG":
            out = out.squeeze(1)
        
        l = loss(out, y)
        l.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()
        
        if i >= temp - 1:
            break

    if (not args.lr_sched_iter):
        scheduler.step()

    return train_loss / temp


def evaluate(context, args, model, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0
    
    if fsd_epoch is None:

        ys, outs, n_eval, n_data = [], [], 0, 0

        with torch.no_grad():
            for i, data in enumerate(loader):
                if transform is not None:
                    x, y, z = data
                    z = z.to(args.device)
                else:
                    x, y = data
                                    
                x, y = x.to(args.device), y.to(args.device)

                out = model(x)

                if isinstance(out, dict):
                    out = out['out']

                if decoder is not None:
                    out = decoder.decode(out).view(x.shape[0], -1)
                    y = decoder.decode(y).view(x.shape[0], -1)
                                    
                if transform is not None:
                    out = transform(out, z)
                    y = transform(y, z)

                if args.dataset[:4] == "DRUG":
                    out = out.squeeze(1)

                outs.append(out)
                ys.append(y)
                n_data += x.shape[0]

                if n_data >= args.eval_batch_size or i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    eval_score += metric(outs, ys).item()
                    n_eval += 1

                    ys, outs, n_data = [], [], 0

            eval_loss /= n_eval
            eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                x, y = loader[ix]
                x, y = x.to(args.device), y.to(args.device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = 1-np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    return eval_loss, eval_score


########################## Helper Funcs ##########################

def save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
        return ep

    else:
        checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
        with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
            save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
            return uuid


def save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats):
    np.save(os.path.join(path, 'hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'embedder_stats.npy'), embedder_stats)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.get_rng_state(),
                # work around for new server
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))


def load_embedder(use_determined, args):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        return os.path.isfile(os.path.join(path, 'state_dict.pt'))
    else:

        info = det.get_cluster_info()
        checkpoint_id = info.latest_checkpoint
        return checkpoint_id is not None

def check_if_continue_training(args):
    """Check if have a check point from result.

    Args:
        args (_type_): _description_

    Returns:
        boolean 
    """
    path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
    if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return False
    return True
        
def load_state(use_determined, args, context, model, optimizer, scheduler, n_train, checkpoint_id=None, test=False, freq=1):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return model, 0, 0, [], [], None
    else:

        if checkpoint_id is None:
            info = det.get_cluster_info()
            checkpoint_id = info.latest_checkpoint
            if checkpoint_id is None:
                return model, 0, 0, [], [], None
        
        checkpoint = client.get_checkpoint(checkpoint_id)
        path = checkpoint.download()

    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    embedder_stats = np.load(os.path.join(path, 'embedder_stats.npy'))
    epochs = freq * (len(train_score) - 1) + 1
    checkpoint_id = checkpoint_id if use_determined else epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    if not test:
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(model_state_dict['scheduler_state_dict'])

        rng_state_dict = torch.load(os.path.join(path, 'rng_state.ckpt'), map_location='cpu')
        # torch.set_rng_state(rng_state_dict['cpu_rng_state'])
        # torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
        # np.random.set_state(rng_state_dict['numpy_rng_state'])
        # random.setstate(rng_state_dict['py_rng_state'])

        if use_determined: 
            try:
                for ep in range(epochs):
                    if ep % freq == 0:
                        context.train.report_training_metrics(steps_completed=(ep + 1) * n_train, metrics={"train loss": train_losses[ep // freq]})
                        context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train, metrics={"val score": train_score[ep // freq]})
            except:
                print("load error")

    return model, epochs, checkpoint_id, list(train_score), list(train_losses), embedder_stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
    parser.add_argument('--log_folder', type= str, default= None, help='[option]path to log folder')
    parser.add_argument('--C_entropy', type= bool, default= False, help='[option]determind Conditional entropy label matching or not')
    parser.add_argument('--second_train', type= bool, default= False, help='[option]determind second train model or not')
    parser.add_argument('--fm_ep', type= int, default= None, help='Number of feature matching')
    parser.add_argument('--lm_ep', type= int, default= None, help='Number of label matching')
    parser.add_argument('--seed', type= int, default= None, help='[optional]Seed for training')
    parser.add_argument('--pde', type= bool, default= False, help='[optional]PDE dataset or not')
    parser.add_argument('--id', type= int, default= None, help='[optional]Id of experiment')
    
    
    args = parser.parse_args()
    fm_ep = args.fm_ep
    lm_ep = args.lm_ep
    seed = args.seed
    id = args.id
    root_dataset = args.root_dataset
    log_folder = args.log_folder
    C_entropy = args.C_entropy
    second_train = args.second_train

    pde = args.pde
    ############################################
    if args.config is not None:     
        import yaml
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
            args = SimpleNamespace(**config['hyperparameters'])
            
            if (fm_ep is not None): 
                args.embedder_epochs = fm_ep
            if (lm_ep is not None):
                args.label_epochs = lm_ep
            if (seed is not None):
                args.seed = seed

            args.experiment_id = random.randint(1000, 9999)
            if id is not None:
                args.experiment_id = id
            args.finetune_method = args.finetune_method + 'FM_' + str(args.embedder_epochs) + '_CE_' + str(args.label_epochs) + '_seed_' + str(args.seed) 
             
            setattr(args, 'C_entropy', C_entropy)
            setattr(args, 'pde', pde)
            if not hasattr(args, 'label_epochs'):
                setattr(args, 'label_epochs', 1)         
            main(False, args, DatasetRoot= root_dataset, log_folder= log_folder, second_train= second_train)
    else:
        print("Config for training not found...")
    