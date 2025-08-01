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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def main(use_determined ,args,info=None, context=None, DatasetRoot= None, log_folder = None, second_train = False):

    ############## Init log file and set seed ###############################
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    
    #################### Load config ####################################
    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    print("current configs: ", args)
    
    ####### determind type of backbone ##################################
    Roberta = True if len(sample_shape) == 3 else False
    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    ########## init tgt_model ###########################################
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()
    
    
    if Roberta is not True: 
        print("2D task...")    
        src_num_classes = 10
        ###### get tgt data 
        tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)  
        transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        ###### get src feature
        src_model, src_train_feature = get_pretrain_model2D_feature(args,root,sample_shape,num_classes,src_num_classes)
        src_feats_visualize = src_train_feature.tensors[0]
        ###### Do feature matching 
        tgt_model = feature_matching_tgt_model(args,root, tgt_model,src_train_feature)
        tgt_feats_visualize_before = get_feature_extract_from_model(tgt_model, tgt_train_loader, transform)
        #visualize_feature_save(src_feats_visualize,tgt_feats_visualize_before)
        ###### get tgt feature before label-matching 
        print("Do fast label matching...")
        src_model = label_matching_by_entropy(args,root, src_model, tgt_model.embedder, num_classes, model_type="2D")
        ####### Init body model and embedder 
        tgt_model.model.swin.encoder = src_model.model.swin.encoder
        tgt_model.model.swin.embeddings= src_model.model.swin.embeddings
        tgt_model.embedder = src_model.embedder
        del src_model
        ###### get tgt feature after-label-matching 
        tgt_feats_visualize_after = get_feature_extract_from_model(tgt_model, tgt_train_loader, transform)
        visualize_feature_save(tgt_feats_visualize_before,tgt_feats_visualize_after, name = "after")


        
        
    else:
        print("1D task...")
        #### get source train dataset 
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
       
    

def visualize_feature(tgt_feature, src_feature):
    """
    Visualize target and source features using 2D and 3D t-SNE.
    
    Parameters:
    tgt_feature : numpy.ndarray
        Target feature array (n_samples, n_features)
    src_feature : numpy.ndarray
        Source feature array (n_samples, n_features)
    
    Returns:
    None (displays 2D and 3D t-SNE plots)
    """
    # Combine features for t-SNE
    combined_features = np.vstack((tgt_feature, src_feature))
    n_tgt = tgt_feature.shape[0]
    n_src = src_feature.shape[0]
    
    # Create labels for visualization
    labels = np.array(['Target'] * n_tgt + ['Source'] * n_src)
    
    # 2D t-SNE
    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_2d_results = tsne_2d.fit_transform(combined_features)
    
    # 3D t-SNE
    tsne_3d = TSNE(n_components=3, random_state=42)
    tsne_3d_results = tsne_3d.fit_transform(combined_features)
    
    # Plot 2D t-SNE
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(tsne_2d_results[mask, 0], tsne_2d_results[mask, 1], 
                   label=label, alpha=0.6)
    plt.title('2D t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    
    # Plot 3D t-SNE
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(122, projection='3d')
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(tsne_3d_results[mask, 0], tsne_3d_results[mask, 1], 
                  tsne_3d_results[mask, 2], label=label, alpha=0.6)
    ax.set_title('3D t-SNE Visualization')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return 0


def visualize_feature_save(tgt_feature, src_feature, output_dir=".", name= "before"):
    """
    Visualize target and source features using 2D and 3D t-SNE, saving plots to files.
    
    Parameters:
    tgt_feature : numpy.ndarray
        Target feature array (n_samples, n_features)
    src_feature : numpy.ndarray
        Source feature array (n_samples, n_features)
    output_dir : str
        Directory to save the output plots (default: current directory)
    
    Returns:
    None (saves 2D and 3D t-SNE plots as PNG files)
    """
    # Combine features for t-SNE
    if isinstance(tgt_feature, torch.Tensor):
        tgt_feature = tgt_feature.detach().cpu().numpy()
    if isinstance(src_feature, torch.Tensor):
        src_feature = src_feature.detach().cpu().numpy()
    combined_features = np.vstack((tgt_feature, src_feature))
    n_tgt = tgt_feature.shape[0]
    n_src = src_feature.shape[0]
    
    # Create labels for visualization
    labels = np.array(['Target'] * n_tgt + ['Source'] * n_src)
    
    # 2D t-SNE
    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_2d_results = tsne_2d.fit_transform(combined_features)
    
    # 3D t-SNE
    tsne_3d = TSNE(n_components=3, random_state=42)
    tsne_3d_results = tsne_3d.fit_transform(combined_features)
    
    # Plot 2D t-SNE
    plt.figure(figsize=(10, 5))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(tsne_2d_results[mask, 0], tsne_2d_results[mask, 1], 
                   label=label, alpha=0.6)
    plt.title('2D t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.savefig(f"{output_dir}/{name}_tsne_2d.png", bbox_inches='tight')
    plt.close()
    
    # Plot 3D t-SNE
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(tsne_3d_results[mask, 0], tsne_3d_results[mask, 1], 
                  tsne_3d_results[mask, 2], label=label, alpha=0.6)
    ax.set_title('3D t-SNE Visualization')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.legend()
    plt.savefig(f"{output_dir}/{name}_tsne_3d.png", bbox_inches='tight')
    plt.close()

    return 0


def get_feature_extract_from_model(tgt_model, tgt_loader,transform, maxsamples = 500):
    tgt_model.output_raw = True
    tgt_model.eval()
    feats = []
    datanum = 0
    for j, data in enumerate(tgt_loader):
            
            if transform is not None:
                x, y, z = data
            else:
                x, y = data 
                
            x = x.to(args.device)
            out = tgt_model(x)
            
            feats.append(out)
            datanum += x.shape[0]
                
            if datanum > maxsamples:
                  break


    feats = torch.cat(feats, 0).mean(1)
    tgt_model.output_raw = False
    tgt_model.train()
    return feats          


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
    parser.add_argument('--pde', type= bool, default= False, help='[optional]PDE dataset or not')
    
    
    
    args = parser.parse_args()
    root_dataset = args.root_dataset
    pde = args.pde
    ############################################
    if args.config is not None:     
        import yaml
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
            args = SimpleNamespace(**config['hyperparameters'])
          
            setattr(args, 'pde', pde)   
            main(False, args, DatasetRoot= root_dataset)
    else:
        print("Config for training not found...")
    