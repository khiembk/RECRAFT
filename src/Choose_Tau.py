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
    
         
    # print("final model: ", tgt_model)

    # ######### load tgt dataset 
    # print("load tgt dataset ...")
    # train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    # metric, compare_metrics = get_metric(root, args.dataset)
    # decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    # transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    # ###############################################################################################
    
    # print("load dic if model was trained ...")
    # tgt_model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, tgt_model, None, None, n_train, freq=args.validation_freq, test=True)
    # # embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    # offset = 0 if ep_start == 0 else 1
    # args, tgt_model, optimizer, scheduler = get_optimizer_scheduler(args, tgt_model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    # train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs
   
    # if args.device == 'cuda':
    #     tgt_model.cuda()
    #     try:
    #         loss.cuda()
    #     except:
    #         pass
    #     if decoder is not None:
    #         decoder.cuda()

    # print("\n------- Experiment Summary --------")
    # print("id:", args.experiment_id)
    # print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer["params"]["lr"])
    # print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    # print("finetune method:", args.finetune_method)
    # print("all param count:", count_params(tgt_model),)
    # logging.info("all param count: %d", count_params(tgt_model))
    # print("trainabel params count: %d  ",count_trainable_params(tgt_model))
    # logging.info("trainabel params count:  %d  ",count_trainable_params(tgt_model))
    # # print("print model")
    # # print(model)
    # logging.info(f"Model Structure:\n{tgt_model}")
    # ###### load state optimizer
    # if second_train:
    #     tgt_model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, tgt_model, None, None, n_train, freq=args.validation_freq, test = True)
    # else:
    #     tgt_model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, tgt_model, optimizer, scheduler, n_train, freq=args.validation_freq)
    # #embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    # train_time = []
    # embedder_stats = []
    # print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")

    # for ep in range(ep_start, args.epochs + args.predictor_epochs):
    #     if not train_full and ep >= args.predictor_epochs:
    #         args, tgt_model, optimizer, scheduler = get_optimizer_scheduler(args, tgt_model, module=None, n_train=n_train)
    #         train_full = True

    #     time_start = default_timer()

    #     train_loss = train_one_epoch(context, args, tgt_model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform)
    #     train_time_ep = default_timer() -  time_start 

    #     if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
    #         val_loss, val_score = evaluate(context, args, tgt_model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

    #         train_losses.append(train_loss)
    #         train_score.append(val_score)
    #         train_time.append(train_time_ep)

    #         print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))
    #         logging.info(
    #         "[train %s %d %.6f] time elapsed: %.4f\ttrain loss: %.4f\tval loss: %.4f\tval score: %.4f\tbest val score: %.4f",
    #         "full" if ep >= args.predictor_epochs else "predictor",ep,optimizer.param_groups[0]['lr'], train_time[-1], train_loss,
    #          val_loss, val_score,compare_metrics(train_score))
    #         wandb.log({
    #         "epoch": ep,
    #         "time_elapsed": train_time[-1],
    #         "train_loss": train_loss,
    #         "val_loss": val_loss,
    #         "val_score": val_score,
    #         "best_val_score": compare_metrics(train_score)
    #         })
    #         if use_determined :
    #             id_current = save_state(use_determined, args, context, tgt_model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
    #             try:
    #                 context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
    #                 context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
    #             except:
    #                 pass
                    
    #         if compare_metrics(train_score) == val_score:
    #             if not use_determined :
    #                 print("save state at epoch ep: ", ep)
    #                 id_current = save_state(use_determined, args, context, tgt_model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
    #             id_best = id_current
            

    #     if ep == args.epochs + args.predictor_epochs - 1:
    #         print("\n------- Start Test --------")
    #         test_scores = []
    #         test_model = tgt_model
    #         test_time_start = default_timer()
    #         test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
    #         test_time_end = default_timer()
    #         test_scores.append(test_score)

    #         print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
    #         logging.info("[test last]\ttime elapsed: %.4f\ttest loss: %.4f\ttest score: %.4f",test_time_end - test_time_start,test_loss,test_score)

    #         test_model, _, _, _, _, _ = load_state(use_determined, args, context, test_model, optimizer, scheduler, n_train, id_best, test=True)
    #         test_time_start = default_timer()
    #         test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
    #         test_time_end = default_timer()
    #         test_scores.append(test_score)

    #         print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
    #         logging.info("[test best-validated]\ttime elapsed: %.4f\ttest loss: %.4f\ttest score: %.4f" % (test_time_end - test_time_start, test_loss, test_score))
            

    #         if use_determined:
    #             checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
    #             with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
    #                 np.save(os.path.join(path, 'test_score.npy'), test_scores)
    #         else:
    #             path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
    #             np.save(os.path.join(path, 'test_score.npy'), test_scores)

           
    #     if use_determined and context.preempt.should_preempt():
    #         print("paused")
    #         return

   
    # wandb.finish()    






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
    