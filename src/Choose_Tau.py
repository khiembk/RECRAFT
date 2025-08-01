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
from newEmbedder import get_pretrain_model2D_feature_with_tau, wrapper1D, wrapper2D, feature_matching_tgt_model,get_src_train_dataset_1Dmodel
from test_model import get_src_predictor1D
import wandb
from datetime import datetime
from newEmbedder import label_matching_by_entropy, label_matching_by_conditional_entropy


def main(use_determined ,args,info=None, context=None, DatasetRoot= None, log_folder = None, second_train = False):

    
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
    
    
   
    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    
    
    #################### Load config 
    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    print("current configs: ", args)
    
    ####### determind type of backbone
    Roberta = True if len(sample_shape) == 3 else False
    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    ########## init tgt_model
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()
    
    
    if Roberta is not True: 
        print("2D task...")
    ######### config for testing 
        
        src_num_classes = 10  
    ######### get src_model and src_feature
        src_model, src_train_dataset = get_pretrain_model2D_feature_with_tau(args,root,sample_shape,num_classes,src_num_classes)
    
    # ######### feature matching for tgt_model.
    #     tgt_model = feature_matching_tgt_model(args,root, tgt_model,src_train_dataset)
    #     del src_train_dataset
    # ######### label matching for src_model.
    #     if args.C_entropy == False:
    #         print("Do fast label matching...")
    #         src_model = label_matching_by_entropy(args,root, src_model, tgt_model.embedder, num_classes, model_type="2D")
    #     else:
    #         print("Do conditional label matching...")
    #         src_model = label_matching_by_conditional_entropy(args,root, src_model, tgt_model.embedder, num_classes, model_type="2D")    
    # ######### fine-tune all tgt_model after feature-label matching.
    #     print("Init tgt_model backbone by src_model...")
    #     tgt_model.model.swin.encoder = src_model.model.swin.encoder
    #     del src_model
    #     set_grad_state(tgt_model.model, True)
    #     set_grad_state(tgt_model.embedder, True)
        
    else:
        print("1D task...")
        # #### get source train dataset 
        # src_train_dataset = get_src_train_dataset_1Dmodel(args,root)
        # ########### feature matching for target model.
        # tgt_model = feature_matching_tgt_model(args,root, tgt_model,src_train_dataset)
        # del src_train_dataset
        # ########## label matching for 1D task
        # print("get src predictor...")
        # src_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
        # src_model.predictor = get_src_predictor1D(args,root)
        # if args.C_entropy == False:
        #     print("Do fast label matching...")
        #     src_model = label_matching_by_entropy(args, root, src_model, tgt_model.embedder, num_classes, model_type="1D")
        # else:
        #     print("Do conditional label matching...")
        #     src_model = label_matching_by_conditional_entropy(args, root, src_model, tgt_model.embedder, num_classes, model_type="1D")    
        # ####################################################################################
        # #Init tgt body model
        # tgt_model.model.encoder = src_model.model.encoder
        # del src_model
        # set_grad_state(tgt_model.model, True)
        # set_grad_state(tgt_model.embedder, True)
        ######################################################
    
         
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    args = parser.parse_args()
    ############################################
    if args.config is not None:     
        import yaml
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
            args = SimpleNamespace(**config['hyperparameters'])
            
            main(False, args)
    else:
        print("Config for training not found...")
    