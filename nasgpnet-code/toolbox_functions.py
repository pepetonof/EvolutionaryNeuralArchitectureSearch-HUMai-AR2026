# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:25:34 2022

@author: josef
"""
import torch.nn as nn
import pygraphviz as pgv
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import stgp

from model.model import BackBone
from objective_functions import evaluate_NoParameters, evaluate_Segmentation, evaluate_complexity
from data.dataloader import loaders

import pickle
import os
import numpy as np
from sklearn.model_selection import KFold
from torchinfo import summary

#%%Helper functions
def _get_metric_keys(metrics):
    """Extract clean metric names."""
    keys = []
    for m in metrics:
        key = str(type(m)).strip('>').strip("'").split('.')[-1]
        keys.append(key)
    return keys


def _init_metric_dict(metric_keys):
    """Initialize empty lists for collecting CV metrics."""
    return {k: [] for k in metric_keys}


def _assign_bad_metrics(cv_metrics, k_folds):
    """Assign bad penalties when the model is too large."""
    for key in cv_metrics.keys():
        if key in ("DiceMetric", "IoUMetric"):
            cv_metrics[key] = [0.0] * k_folds
        else:
            cv_metrics[key] = [10.0] * k_folds


def _append_metrics(cv_metrics, metricsTest):
    """Append means from metricsTest to cv_metrics."""
    for key, vals in metricsTest.items():
        cv_metrics[key].append(np.mean(vals))


def _flatten_cv_metrics(cv_metrics):
    """Flatten structure into a list of mean metrics for output."""
    return [np.mean(v) for v in cv_metrics.values()]


def _save_evaluation_data(
        base_path, fold, dloaders, ind,
        metricsTest, metricsVal, lossAndDice,
        train_set, valid_set, test_set
    ):
    """
    If fold=None, saves to base_path/data/
    If fold=int, saves to base_path/data/fold_{fold}/
    """

    # --------- Path Handling ---------
    if fold is None:
        path_data = f"{base_path}/data"
    else:
        path_data = f"{base_path}/data/fold_{fold}"

    os.makedirs(path_data, exist_ok=True)

    #pickle
    with open(f"{path_data}/metricsTest.pkl", "wb") as f:
        pickle.dump(metricsTest, f)
    with open(f"{path_data}/metricsVal.pkl", "wb") as f:
        pickle.dump(metricsVal, f)
    with open(f"{path_data}/lossAndDice.pkl", "wb") as f:
        pickle.dump(lossAndDice, f)

    #summary statistics
    d = {
        "Height": dloaders.IMAGE_HEIGHT,
        "Width": dloaders.IMAGE_WIDTH,
        "Train_Size": len(dloaders.TRAIN_IMG_DIR),
        "Valid_Size": len(dloaders.VAL_IMG_DIR),
        "Test_Size": len(dloaders.TEST_IMG_DIR),
        "Ind": ind,
    }

    stats = [np.mean, np.median, np.max, np.min, np.std]
    for key, vals in metricsTest.items():
        for s in stats:
            d[f"{key}_{s.__name__}"] = s(vals)

    with open(f"{path_data}/data.txt", 'w', encoding="utf-8") as f:
        for k, v in d.items():
            f.write(f"{k}\n{v}\n")

    #dataset split names
    def _save_set(fname, ds):
        with open(f"{path_data}/{fname}.txt", 'w', encoding="utf-8") as f:
            for k, vals in ds.items():
                f.write(f"\n{k}\n")
                for p in vals:
                    f.write(f"{p.name}\n")

    _save_set("train_set", train_set)
    _save_set("valid_set", valid_set)
    _save_set("test_set",  test_set)


def _run_cv_loop(
        model, ind, metrics, k_folds,
        train_set, valid_set, test_set,
        batch_size, image_height, image_width,
        dataset_type, no_classes_msk,
        nepochs, tolerance, lossfn, lr,
        ruta_ind, device, verbose_train,
        save_model, save_images, save_data
    ):
    """
    Shared Cross validation logic used by evaluation_cv and evaluation_monas.
    Returns:
        cv_metrics: dict of list metrics
        num_folds: number of folds executed
    """

    metric_keys = _get_metric_keys(metrics)
    cv_metrics = _init_metric_dict(metric_keys)

    # Case A: Standard K-Fold
    if isinstance(k_folds, int):
        kfold = KFold(n_splits=k_folds, shuffle=False)
        data_fold = train_set["images"] + valid_set["images"]
        mask_fold = train_set["masks"] + valid_set["masks"]

        folds = []
        for tr, va in kfold.split(data_fold):
            train_fold = {"images":[data_fold[i] for i in tr],
                          "masks":[mask_fold[i] for i in tr]}
            valid_fold = {"images":[data_fold[i] for i in va],
                          "masks":[mask_fold[i] for i in va]}
            folds.append((train_fold, valid_fold, test_set))

    # Case B: Pre-split folds
    else:
        folds = [(train_set[i], valid_set[i], test_set[i]) for i in range(len(train_set))]

    # ----- Main CV Loop -----
    for fold_idx, (train_f, valid_f, test_f) in enumerate(folds, 1):

        dloaders = loaders(train_f, valid_f, test_f,
                           batch_size=batch_size,
                           image_height=image_height,
                           image_width=image_width,
                           dataset_type=dataset_type,
                           no_classes_msk=no_classes_msk)

        metricsTest, metricsVal, lossAndDice = evaluate_Segmentation(
            model, nepochs, tolerance, lossfn, metrics, lr,
            dloaders, device, ruta_ind, verbose_train,
            save_model, save_images, fold=fold_idx
        )

        _append_metrics(cv_metrics, metricsTest)

        if save_data:
            _save_evaluation_data(
                ruta_ind, fold_idx, dloaders, ind,
                metricsTest, metricsVal, lossAndDice,
                train_f, valid_f, test_f
            )

        # reset model weights after fold
        model.initialize_weigths()

    return cv_metrics



#%%Make model
def make_model(ind, in_channels, out_channels, pset):
    """Compile function"""
    func = stgp.compile(expr=ind, pset=pset)
    """Init module: empty sequential module"""
    init_module=[nn.ModuleList(), in_channels]
    """Output of the first block"""
    first_block=func(init_module)
    model=BackBone(first_block, out_channels)
    return model

#%% Evaluate surrogate
def evaluation_surrogate(ind, surrogate_model, 
                         in_channels,
                         out_channels,
                         pset,
                         max_params):

    fit = surrogate_model.predict([ind])[0]
    
    #Generate model
    model = make_model(ind, in_channels, out_channels, pset)
    _, params = evaluate_NoParameters(model, max_params)
    
    # print('Syntax tree:\t', str(ind), params)
    
    return fit, params

#%% Evaluation cv
def evaluation_cv(
        ind, nepochs, tolerance, lossfn, metrics, lr,
        dataset_type, no_classes_msk,
        in_channels, out_channels,
        batch_size, image_height, image_width,
        max_params, w,
        train_set, valid_set, test_set,
        k_folds, pset, ruta,
        device='cpu',
        verbose_train=False,
        save_model=False,
        save_images=False,
        save_data=False,
        limit=100000000
    ):

    ruta_ind = f"{ruta}/{str(ind)}"
    print("Syntax tree:", ind)

    # Build model & compute parameters
    model = make_model(ind, in_channels, out_channels, pset)
    complexity, params = evaluate_NoParameters(model, max_params)
    print("Params:", params)

    # If model too large, penalty
    if params >= limit:
        cv_metrics = _init_metric_dict(_get_metric_keys(metrics))
        _assign_bad_metrics(cv_metrics, k_folds)
        fit = (1 - w) * np.mean(cv_metrics["DiceMetric"]) + w * complexity
        return [fit, params] + _flatten_cv_metrics(cv_metrics)

    cv_metrics = _run_cv_loop(
        model, ind, metrics, k_folds,
        train_set, valid_set, test_set,
        batch_size, image_height, image_width,
        dataset_type, no_classes_msk,
        nepochs, tolerance, lossfn, lr,
        ruta_ind, device, verbose_train,
        save_model, save_images, save_data
    )
    
    del model

    # Compute final fitness
    fit = (1 - w) * np.mean(cv_metrics["DiceMetric"]) + w * complexity

    return [fit, params] + _flatten_cv_metrics(cv_metrics)


#%%Evaluation monas
def evaluation_monas(
        ind, nepochs, tolerance, lossfn, metrics, lr,
        dataset_type, no_classes_msk,
        in_channels, out_channels,
        batch_size, image_height, image_width,
        train_set, valid_set, test_set,
        k_folds, pset, ruta,
        device='cpu',
        verbose_train=False,
        save_model=False, save_images=False,
        save_data=False, limit=100000000
    ):
    
    ruta_ind = f"{ruta}/{str(ind)}"
    # print("Syntax tree:", ind)

    # Multi-objective complexity (log scaling version)
    model = make_model(ind, in_channels, out_channels, pset)
    params = evaluate_complexity(model)
    # print("Params:", params)

    # Too large?
    if params >= limit:
        cv_metrics = _init_metric_dict(_get_metric_keys(metrics))
        _assign_bad_metrics(cv_metrics, k_folds)
        dice_loss = 1 - np.mean(cv_metrics["DiceMetric"])
        return (dice_loss, np.log10(params))
    
    cv_metrics = _run_cv_loop(
        model, ind, metrics, k_folds,
        train_set, valid_set, test_set,
        batch_size, image_height, image_width,
        dataset_type, no_classes_msk,
        nepochs, tolerance, lossfn, lr,
        ruta_ind, device, verbose_train,
        save_model, save_images, save_data
    )

    del model
    
    dice_loss = 1 - np.mean(cv_metrics["DiceMetric"])
    # return (dice_loss, np.log10(params))
    return [dice_loss, np.log10(params)] + _flatten_cv_metrics(cv_metrics)



#%%evaluation no cv
def evaluation(
        ind, nepochs, tolerance, lossfn, metrics, lr,
        dataset_type, no_classes_msk,
        in_channels, out_channels,
        batch_size, image_height, image_width,
        max_params, w,
        train_set, valid_set, test_set,
        pset, ruta,
        device='cpu',
        verbose_train=False,
        save_model=False, save_images=False,
        save_data=False,
        limit=100000000
    ):

    ruta_ind = f"{ruta}/{str(ind)}"

    # Build model
    model = make_model(ind, in_channels, out_channels, pset)
    complexity, params = evaluate_NoParameters(model, max_params)

    metric_keys = _get_metric_keys(metrics)
    metric_vals = {k: 0.0 for k in metric_keys}

    # Too large model?
    if params >= limit:
        for k in metric_vals:
            metric_vals[k] = 0.0 if k in ("DiceMetric", "IoUMetric") else 10.0
        fit = (1 - w) * metric_vals["DiceMetric"] + w * complexity
        return [fit, params] + list(metric_vals.values())

    # Loaders
    dloaders = loaders(train_set, valid_set, test_set,
                       batch_size=batch_size,
                       image_height=image_height, image_width=image_width,
                       dataset_type=dataset_type,
                       no_classes_msk=no_classes_msk)

    # Train/eval
    metricsTest, metricsVal, lossAndDice = evaluate_Segmentation(
        model, nepochs, tolerance, lossfn, metrics,
        lr, dloaders, device, ruta_ind,
        verbose_train, save_model, save_images,
        fold=None
    )

    # Average metrics
    for key, vals in metricsTest.items():
        metric_vals[key] = np.mean(vals)

    # Optional saving
    if save_data:
        
        _save_evaluation_data(ruta_ind, None, dloaders, ind,
                                metricsTest, metricsVal, lossAndDice,
                                train_set, valid_set, test_set)

    fit = (1 - w) * metric_vals["DiceMetric"] + w * complexity
    del model
    return [fit, params] + list(metric_vals.values())








    

# def evaluation_cv(ind, 
#                   nepochs, #For training in original fitness function
#                   tolerance,
#                   lossfn,
#                   metrics,
#                   lr, 
#                   dataset_type, #Data2D, Data3D, Data3D22D
#                   no_classes_msk,
#                   in_channels, 
#                   out_channels,
#                   batch_size,
#                   image_height,
#                   image_width,
#                   max_params, w, 
#                   train_set, 
#                   valid_set, 
#                   test_set,
#                   k_folds, #Cross-validation, if False the data is alreadly splitted
#                   pset, 
#                   ruta,
#                   device = 'cpu',  #Device to train the model, cuda device is prefered
#                   verbose_train = False, #Print the training process?
#                   save_model = False, #Save the trained model?
#                   save_images = False, #Save images in the validation and test set?
#                   save_data = False, #Save data about each fold?
#                   limit = 100000000
#                   ):
    
#     #Generate a folder with the string of individual
#     ruta_ind = ruta+"/"+str(ind)
    
#     print('Syntax tree:\t', str(ind))
    
#     #Generate model
#     model = make_model(ind, in_channels, out_channels, pset)
    
#     #Evaluate no of parameters
#     complexity, params = evaluate_NoParameters(model, max_params)
#     print(params)
    
#     #Cross Validation Information
#     cv_metrics = {}
#     for m in metrics:
#         key=str(type(m)).strip('>').strip("'").split('.')[-1]
#         cv_metrics[key]=[]
    
#     #Check the max number of allowed parameters, 
#     #if does not fit in the RAM, just assign an bad metric
#     if params>=limit:
#         for key, value in cv_metrics.items():
#             if key=='DiceMetric' or key=='IoUMetric':
#                 cv_metrics[key]=[0.0]*k_folds
#             else:
#                 cv_metrics[key]=[10.0]*k_folds
        
#     #else, if fits in the RAM continue with k-fold    
#     else:        
#         if type(k_folds)==int:
#             #Evaluate Segmentation Metrics
#             kfold = KFold(n_splits=k_folds, shuffle=False)
        
#             #Merge train and valid. Leave test
#             data_fold = train_set["images"]+valid_set["images"]
#             mask_fold = train_set["masks"]+valid_set["masks"]
        
#             # K-fold Cross Validation model evaluation
#             for fold, (train_idx, valid_idx) in enumerate(kfold.split(data_fold),1):
#                 #Folds
#                 train_set = dict(images = list(map(data_fold.__getitem__, train_idx)), 
#                                  masks = list(map(mask_fold.__getitem__, train_idx)))
#                 valid_set = dict(images = list(map(data_fold.__getitem__, valid_idx)), 
#                                  masks = list(map(mask_fold.__getitem__, valid_idx)))
                
#                 # print('Fold No Split', fold, type(train_set), type(valid_set), type(test_set))
                
#                 #Dataset depending on data
#                 dloaders = loaders(train_set, valid_set, test_set, batch_size=batch_size, 
#                                    image_height=image_height, image_width=image_width,
#                                    dataset_type=dataset_type, no_classes_msk=no_classes_msk,
#                                    )
                
#                 #Evaluate Segentation
#                 # if not surrogate:
#                 metricsTest, metricsVal, lossAndDice  = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics, 
#                                                                               lr, dloaders,
#                                                                               device, ruta_ind, verbose_train,
#                                                                               save_model, save_images, fold=fold)      
#                 # else:
#                 # metricsTest = evaluate_Surrogate(ind, surrogate)
  
#                 for key, value in metricsTest.items():
#                     cv_metrics[key].append(np.mean(value))
                
#                 #Save information related with fold
#                 if save_data:
#                     path_data = f"{ruta_ind}/data/fold_{fold}"
                    
#                     #Save data of metrics in as .pkl file
#                     if not os.path.exists(path_data):
#                         os.makedirs(path_data)
#                     with open(f"{path_data}/metricsTest_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsTest, cp_file)
#                     with open(f"{path_data}/metricsValfold_{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsVal, cp_file)
#                     with open(f"{path_data}/lossAndDice_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(lossAndDice, cp_file)
                    
#                     #Save data as .txt with statistics
#                     d={}
#                     d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
#                              "Width":dloaders.IMAGE_WIDTH, 
#                              "Train_Size":len(dloaders.TRAIN_IMG_DIR),
#                              "Valid_Size":len(dloaders.VAL_IMG_DIR),
#                              "Test_Size": len(dloaders.TEST_IMG_DIR),
#                              "Ind": ind, 
#                              "Ind_Fitness":(1 - w)*np.mean(metricsTest["DiceMetric"]) + w*complexity}
#                     d.update(d_aux)
                    
#                     d_aux={}
#                     statics = [np.mean, np.median, np.max, np.min, np.std]
#                     for key, value in metricsTest.items():
#                         for s in statics:
#                             composed_key=key+'_'+s.__name__
#                             d_aux[composed_key]=s(value)
#                     d.update(d_aux)

#                     model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
#                     summary_model = str(model_stats)
#                     d_aux = {"Summary_Model": summary_model}
#                     d.update(d_aux)
                
#                     with open(f"{path_data}/data_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in d.items(): 
#                             f.write('%s\n%s\n' % (key, value))
                            
#                     #Save the train_set, valid_set and test_set in a .txt files
#                     with open(f"{path_data}/train_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in train_set.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/valid_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in valid_set.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/test_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in test_set.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
  
#                 # cv_add_train.append(add_train)
                
#                 #Print
#                 # if verbose_train:
#                 #     print('DSC_mean for fold %d: %f' % (fold, cv_dices[-1]))
#                 #     print('Iou_mean for fold %d: %f' % (fold, cv_ious[-1]))
#                 #     print('HDS_mean for fold %d: %f' % (fold, cv_hds[-1]))
#                 #     print('HDS95_mean for fold %d: %f' % (fold, cv_hds95[-1]))
#                 #     print('NSD_mean for fold %d: %f' % (fold, cv_nsds[-1]))
                
#                 #Generate model again to reinitialize the weights in each fold
#                 # model.initialize_weigths()
#                 model = make_model(ind, in_channels, out_channels, pset)
            
#         elif k_folds==False:#Ignore fold due the folds are already computed
#             #Evaluate pre computed folds. The paths to files are stored in train_set, test_set, valid_set as a list of dictionaries similar to those used when split is false
#             # a=len(train_set)
#             # print(a)
#             for fold in range(1, len(train_set)+1, 1):
#                 #Split train to obtain valid set:
#                 train_set_fold = train_set[fold-1]
#                 valid_set_fold = valid_set[fold-1]
#                 test_set_fold = test_set[fold-1]
                
#                 # print('Fold Split', fold, type(train_set_fold), type(valid_set_fold), type(test_set_fold))
                
#                 #Dataset depending on data
#                 dloaders = loaders(train_set_fold, valid_set_fold, test_set_fold, batch_size=batch_size, 
#                                    image_height=image_height, image_width=image_width,
#                                    dataset_type=dataset_type, no_classes_msk=no_classes_msk,
#                                    )
#                 # print('ModelDevice', next(model.parameters()).device)
#                 #Evaluate Segentation
#                 metricsTest, metricsVal, lossAndDice  = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics, 
#                                                                               lr, dloaders,
#                                                                               device, ruta_ind, verbose_train,
#                                                                               save_model, save_images,
#                                                                               fold=fold)
                
#                 for key, value in metricsTest.items():
#                     cv_metrics[key].append(np.mean(value))
                
#                 #Save information about fold
#                 if save_data:
#                     path_data = f"{ruta_ind}/data/fold_{fold}"
#                     if not os.path.exists(path_data):
#                         os.makedirs(path_data)
#                     with open(f"{path_data}/metricsTest_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsTest, cp_file)
#                     with open(f"{path_data}/metricsVal_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsVal, cp_file)
#                     with open(f"{path_data}/lossAndDice_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(lossAndDice, cp_file)
                        
#                     #Save data as .txt with statistics
#                     d={}
#                     d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
#                              "Width":dloaders.IMAGE_WIDTH, 
#                              "Train_Size":len(dloaders.TRAIN_IMG_DIR),
#                              "Valid_Size":len(dloaders.VAL_IMG_DIR),
#                              "Test_Size": len(dloaders.TEST_IMG_DIR),
#                              "Ind":ind, 
#                              "Ind_Fitness":(1 - w)*np.mean(metricsTest["DiceMetric"]) + w*complexity}
#                     d.update(d_aux)
                    
#                     d_aux={}
#                     statics = [np.mean, np.median, np.max, np.min, np.std]
#                     for key, value in metricsTest.items():
#                         for s in statics:
#                             composed_key=key+'_'+s.__name__
#                             d_aux[composed_key]=s(value)
#                     d.update(d_aux)
                    
#                     model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
#                     summary_model = str(model_stats)
#                     d_aux = {"Summary_Model": summary_model}
#                     d.update(d_aux)
                    
#                     with open(f"{path_data}/data_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in d.items(): 
#                             f.write('%s\n%s\n' % (key, value))
                    
#                     #Save the train_set, valid_set and test_set in a .txt files
#                     with open(f"{path_data}/train_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in train_set_fold.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/valid_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in valid_set_fold.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/test_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in test_set_fold.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                
#                 #Generate model again to reinitialize the weights in each fold
#                 model = make_model(ind, in_channels, out_channels, pset)
#                 # model.initialize_weigths()

#     #Fitness as lienar combination of mean dice and the number of parameters
#     fit = (1 - w)*np.mean(cv_metrics["DiceMetric"]) + w*complexity
    
#     #Fitness as minimization
#     # fit = (1-w)*np.mean(cv_dices_loss) + w*(params/max_params)
#     # alpha = 0.25
#     # beta = 0.25
#     # DSC_loss_train = alpha*np.mean(1 - np.array(cv_dices_t))
#     # DSC_loss_val   = np.mean(1 - np.array(cv_dices))
#     # ADD_train      = beta*np.mean(np.array(cv_add_train))
    
#     # f1 = DSC_loss_val + beta*ADD_train + alpha*DSC_loss_train
#     # f2 = params/max_params
#     # print(f1,f2)
#     # fit= (1-w)*f1 + w*f2

#     # print(f1, DSC_loss_val, DSC_loss_train, DSC_loss_train*alpha, ADD_train, f2, params, max_params)
#     #np.mean(cv_dices_t),
#     del model
#     return [fit, params] + [np.mean(a) for a in cv_metrics.values()]
    #return fit, np.mean(cv_dices), np.mean(cv_ious), np.mean(cv_hds), np.mean(cv_hds95), np.mean(cv_nsds), params #hd95, nds
    
    



# def evaluation_monas(ind, 
#                   nepochs, #For training in original fitness function
#                   tolerance,
#                   lossfn,
#                   metrics,
#                   lr, 
#                   dataset_type, #Data2D, Data3D, Data3D22D
#                   no_classes_msk,
#                   in_channels, 
#                   out_channels,
#                   batch_size,
#                   image_height,
#                   image_width,
#                   max_params, w, 
#                   train_set, 
#                   valid_set, 
#                   test_set,
#                   k_folds, #Cross-validation, if False the data is alreadly splitted
#                   pset, 
#                   ruta,
#                   device = 'cpu',  #Device to train the model, cuda device is prefered
#                   verbose_train = False, #Print the training process?
#                   save_model = False, #Save the trained model?
#                   save_images = False, #Save images in the validation and test set?
#                   save_data = False, #Save data about each fold?
#                   limit = 100000000
#                   ):
    
#     #Generate a folder with the string of individual
#     ruta_ind = ruta+"/"+str(ind)
    
#     print('Syntax tree:\t', str(ind))
    
#     #Generate model
#     model = make_model(ind, in_channels, out_channels, pset)
    
#     #Evaluate no of parameters
#     # complexity, params = evaluate_NoParameters(model, max_params)
#     params = evaluate_complexity(model)
#     print(params)
    
#     #Cross Validation Information
#     cv_metrics = {}
#     for m in metrics:
#         key=str(type(m)).strip('>').strip("'").split('.')[-1]
#         cv_metrics[key]=[]
    
#     #Check the max number of allowed parameters, 
#     #if does not fit in the RAM, just assign an bad metric
#     if params>=limit:
#         for key, value in cv_metrics.items():
#             if key=='DiceMetric' or key=='IoUMetric':
#                 cv_metrics[key]=[0.0]*k_folds
#             else:
#                 cv_metrics[key]=[10.0]*k_folds
        
#     #else, if fits in the RAM continue with k-fold    
#     else:        
#         if type(k_folds)==int:
#             #Evaluate Segmentation Metrics
#             kfold = KFold(n_splits=k_folds, shuffle=False)
        
#             #Merge train and valid. Leave test
#             data_fold = train_set["images"]+valid_set["images"]
#             mask_fold = train_set["masks"]+valid_set["masks"]
        
#             # K-fold Cross Validation model evaluation
#             for fold, (train_idx, valid_idx) in enumerate(kfold.split(data_fold),1):
#                 #Folds
#                 train_set = dict(images = list(map(data_fold.__getitem__, train_idx)), 
#                                  masks = list(map(mask_fold.__getitem__, train_idx)))
#                 valid_set = dict(images = list(map(data_fold.__getitem__, valid_idx)), 
#                                  masks = list(map(mask_fold.__getitem__, valid_idx)))
                
#                 # print('Fold No Split', fold, type(train_set), type(valid_set), type(test_set))
                
#                 #Dataset depending on data
#                 dloaders = loaders(train_set, valid_set, test_set, batch_size=batch_size, 
#                                    image_height=image_height, image_width=image_width,
#                                    dataset_type=dataset_type, no_classes_msk=no_classes_msk,
#                                    )
                
#                 #Evaluate Segentation
#                 # if not surrogate:
#                 metricsTest, metricsVal, lossAndDice  = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics, 
#                                                                               lr, dloaders,
#                                                                               device, ruta_ind, verbose_train,
#                                                                               save_model, save_images, fold=fold)      
#                 # else:
#                 # metricsTest = evaluate_Surrogate(ind, surrogate)
  
#                 for key, value in metricsTest.items():
#                     cv_metrics[key].append(np.mean(value))
                
#                 #Save information related with fold
#                 if save_data:
#                     path_data = f"{ruta_ind}/data/fold_{fold}"
                    
#                     #Save data of metrics in as .pkl file
#                     if not os.path.exists(path_data):
#                         os.makedirs(path_data)
#                     with open(f"{path_data}/metricsTest_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsTest, cp_file)
#                     with open(f"{path_data}/metricsValfold_{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsVal, cp_file)
#                     with open(f"{path_data}/lossAndDice_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(lossAndDice, cp_file)
                    
#                     #Save data as .txt with statistics
#                     d={}
#                     d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
#                              "Width":dloaders.IMAGE_WIDTH, 
#                              "Train_Size":len(dloaders.TRAIN_IMG_DIR),
#                              "Valid_Size":len(dloaders.VAL_IMG_DIR),
#                              "Test_Size": len(dloaders.TEST_IMG_DIR),
#                              "Ind": ind, 
#                              #"Ind_Fitness":(1 - w)*np.mean(metricsTest["DiceMetric"]) + w*complexity
#                              }
#                     d.update(d_aux)
                    
#                     d_aux={}
#                     statics = [np.mean, np.median, np.max, np.min, np.std]
#                     for key, value in metricsTest.items():
#                         for s in statics:
#                             composed_key=key+'_'+s.__name__
#                             d_aux[composed_key]=s(value)
#                     d.update(d_aux)

#                     model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
#                     summary_model = str(model_stats)
#                     d_aux = {"Summary_Model": summary_model}
#                     d.update(d_aux)
                
#                     with open(f"{path_data}/data_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in d.items(): 
#                             f.write('%s\n%s\n' % (key, value))
                            
#                     #Save the train_set, valid_set and test_set in a .txt files
#                     with open(f"{path_data}/train_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in train_set.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/valid_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in valid_set.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/test_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in test_set.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
  
#                 #Generate model again to reinitialize the weights in each fold
#                 # model.initialize_weigths()
#                 model = make_model(ind, in_channels, out_channels, pset)
            
#         elif k_folds==False:#Ignore fold due the folds are already computed
#             #Evaluate pre computed folds. The paths to files are stored in train_set, test_set, valid_set as a list of dictionaries similar to those used when split is false
#             # a=len(train_set)
#             # print(a)
#             for fold in range(1, len(train_set)+1, 1):
#                 #Split train to obtain valid set:
#                 train_set_fold = train_set[fold-1]
#                 valid_set_fold = valid_set[fold-1]
#                 test_set_fold = test_set[fold-1]
                
#                 # print('Fold Split', fold, type(train_set_fold), type(valid_set_fold), type(test_set_fold))
                
#                 #Dataset depending on data
#                 dloaders = loaders(train_set_fold, valid_set_fold, test_set_fold, batch_size=batch_size, 
#                                    image_height=image_height, image_width=image_width,
#                                    dataset_type=dataset_type, no_classes_msk=no_classes_msk,
#                                    )
#                 # print('ModelDevice', next(model.parameters()).device)
#                 #Evaluate Segentation
#                 metricsTest, metricsVal, lossAndDice  = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics, 
#                                                                               lr, dloaders,
#                                                                               device, ruta_ind, verbose_train,
#                                                                               save_model, save_images,
#                                                                               fold=fold)
                
#                 for key, value in metricsTest.items():
#                     cv_metrics[key].append(np.mean(value))
                
#                 #Save information about fold
#                 if save_data:
#                     path_data = f"{ruta_ind}/data/fold_{fold}"
#                     if not os.path.exists(path_data):
#                         os.makedirs(path_data)
#                     with open(f"{path_data}/metricsTest_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsTest, cp_file)
#                     with open(f"{path_data}/metricsVal_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(metricsVal, cp_file)
#                     with open(f"{path_data}/lossAndDice_fold{fold}.pkl", "wb") as cp_file:
#                         pickle.dump(lossAndDice, cp_file)
                        
#                     #Save data as .txt with statistics
#                     d={}
#                     d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
#                              "Width":dloaders.IMAGE_WIDTH, 
#                              "Train_Size":len(dloaders.TRAIN_IMG_DIR),
#                              "Valid_Size":len(dloaders.VAL_IMG_DIR),
#                              "Test_Size": len(dloaders.TEST_IMG_DIR),
#                              "Ind":ind, 
#                              # "Ind_Fitness":(1 - w)*np.mean(metricsTest["DiceMetric"]) + w*complexity
#                              }
#                     d.update(d_aux)
                    
#                     d_aux={}
#                     statics = [np.mean, np.median, np.max, np.min, np.std]
#                     for key, value in metricsTest.items():
#                         for s in statics:
#                             composed_key=key+'_'+s.__name__
#                             d_aux[composed_key]=s(value)
#                     d.update(d_aux)
                    
#                     model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
#                     summary_model = str(model_stats)
#                     d_aux = {"Summary_Model": summary_model}
#                     d.update(d_aux)
                    
#                     with open(f"{path_data}/data_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in d.items(): 
#                             f.write('%s\n%s\n' % (key, value))
                    
#                     #Save the train_set, valid_set and test_set in a .txt files
#                     with open(f"{path_data}/train_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in train_set_fold.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/valid_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in valid_set_fold.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                                        
#                     with open(f"{path_data}/test_set_fold{fold}.txt", 'w', encoding="utf-8") as f: 
#                         for key, value in test_set_fold.items(): 
#                             f.write('\n%s\n' % (key))
#                             for name in [path.name for path in value]:
#                                 f.write('%s\n' % (name))
                
#                 #Generate model again to reinitialize the weights in each fold
#                 model = make_model(ind, in_channels, out_channels, pset)
#                 # model.initialize_weigths()

#     dice_loss = 1-np.mean(cv_metrics["DiceMetric"])
    
#     return (dice_loss, np.log10(params))
    
    
# def evaluation(ind,
#                 nepochs,
#                 tolerance,
#                 lossfn,
#                 metrics,
#                 lr,
#                 dataset_type,
#                 no_classes_msk,
#                 in_channels, 
#                 out_channels,
#                 batch_size,
#                 image_height,
#                 image_width,
#                 max_params, w,
#                 train_set,
#                 valid_set,
#                 test_set,
#                 pset,
#                 ruta,
#                 device = 'cpu',
#                 verbose_train = False, 
#                 save_model = False,
#                 save_images = False,
#                 save_data = False,
#                 limit=100000000,
#                 ):
    
#     #Change ruta with name of the str(ind)
#     ruta = ruta+"/"+str(ind)
    
#     #Generate model
#     model = make_model(ind, in_channels, out_channels, pset)
    
#     #Evaluate no of parameters
#     complexity, params = evaluate_NoParameters(model, max_params)
    
#     segmentation_metrics = {}
#     for m in metrics:
#         key=str(type(m)).strip('>').strip("'").split('.')[-1]
#         segmentation_metrics[key]=0.0
    
#     #Check the max number of allowed parameters, 
#     #if does not fit in the RAM, just assign an bad fit
#     if params >= limit:
#         for key, value in segmentation_metrics:#metrics:
#             aux = str(type(m)).strip('>').strip("'").split('.')[-1]
#             if aux == 'DiceMetric' or aux == 'IouMetric':
#                 # setattr(ind.m._get_name(), 0.0) #asigna un mal fitness
#                 segmentation_metrics[key] = 0.0
#             else:
#                 # setattr(ind.m._get_name(), 10.0)
#                 segmentation_metrics[key] = 10.0
            
#     #else, if fits in the RAM continue with train and validation
#     else:
        
#         #Dataloaders from train_set, valid_set, test_set
#         dloaders = loaders(train_set, valid_set, test_set, batch_size=batch_size, 
#                             image_height=image_height, image_width=image_width, 
#                             dataset_type=dataset_type, no_classes_msk=no_classes_msk,
#                             )
        
#         #Evaluate Segmentation Metrics
#         metricsTest, metricsVal, lossAndDice = evaluate_Segmentation(model, nepochs, tolerance, lossfn, metrics,
#                                                                      lr, dloaders, 
#                                                                      device, ruta, verbose_train,
#                                                                      save_model, save_images, fold=None)
        
#         for key, value in metricsTest.items():
#             segmentation_metrics[key] = np.mean(value)
        
#         if save_data:
#             path_data = f"{ruta}/data/"
            
#             #Save data of metrics in as .pkl file
#             if not os.path.exists(path_data):
#                 os.makedirs(path_data)
#             with open(f"{path_data}/metricsTest.pkl", "wb") as cp_file:
#                 pickle.dump(metricsTest, cp_file)
#             with open(f"{path_data}/metricsVal.pkl", "wb") as cp_file:
#                 pickle.dump(metricsVal, cp_file)
#             with open(f"{path_data}/lossAndDice.pkl", "wb") as cp_file:
#                 pickle.dump(lossAndDice, cp_file)
            
#             #Save data as .txt with statistics
#             d={}
#             d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
#                      "Width":dloaders.IMAGE_WIDTH, 
#                      "Train_Size":len(train_set["images"]),#len(dloaders.TRAIN_IMG_DIR),
#                      "Valid_Size":len(train_set["images"]),#len(dloaders.VAL_IMG_DIR),
#                      "Test_Size": len(train_set["images"]),#len(dloaders.TEST_IMG_DIR),
#                      "Ind": ind, 
#                      "Ind_Fitness":(1 - w)*np.mean(metricsTest["DiceMetric"]) + w*complexity}
#             d.update(d_aux)
            
#             d_aux={}
#             statics = [np.mean, np.median, np.max, np.min, np.std]
#             for key, value in metricsTest.items():
#                 for s in statics:
#                     composed_key=key+'_'+s.__name__
#                     d_aux[composed_key]=s(value)
#             d.update(d_aux)

#             model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
#             summary_model = str(model_stats)
#             d_aux = {"Summary_Model": summary_model}
#             d.update(d_aux)
        
#             with open(f"{path_data}/data.txt", 'w', encoding="utf-8") as f: 
#                 for key, value in d.items(): 
#                     f.write('%s\n%s\n' % (key, value))
                    
#             #Save the train_set, valid_set and test_set in a .txt files
#             with open(f"{path_data}/train_set.txt", 'w', encoding="utf-8") as f: 
#                 for key, value in train_set.items(): 
#                     f.write('\n%s\n' % (key))
#                     for name in [path.name for path in value]:
#                         f.write('%s\n' % (name))
                                
#             with open(f"{path_data}/valid_set.txt", 'w', encoding="utf-8") as f: 
#                 for key, value in valid_set.items(): 
#                     f.write('\n%s\n' % (key))
#                     for name in [path.name for path in value]:
#                         f.write('%s\n' % (name))
                                
#             with open(f"{path_data}/test_set.txt", 'w', encoding="utf-8") as f: 
#                 for key, value in test_set.items(): 
#                     f.write('\n%s\n' % (key))
#                     for name in [path.name for path in value]:
#                         f.write('%s\n' % (name))

#     fit = (1 - w)*segmentation_metrics["DiceMetric"] + w*complexity
    
#     return [fit, params] + list(segmentation_metrics.values())
        

# def evaluationMO(ind, nepochs, lossfn, lr,
#                  max_params,
                
#                  loaders, pset, device, ruta, verbose_train):
    
#     # """Make model"""
#     in_channels = loaders.IN_CHANNELS
#     model = make_model(ind, in_channels, pset)
    
#     #Evaluate no of parameters
#     complexity, params = evaluate_NoParameters(model, loaders.IN_CHANNELS, max_params, pset)
#     # print(params)
    
#     #Exceed on COVID
#     #121219529:#680810886:#680810886 #mpool(cat(dCon(conv(mod, 32, 5, 5, 2), 0.7), dCon(sconv(conv(conv(conv(sconv(mod, 32, 5, 7, 1), 16, 7, 3, 2), 32, 7, 5, 2), 32, 7, 7, 1), 8, 5, 7, 1), 0.8)))
#     if params<125000000:
    
#         #Evaluate Segmentation Metrics
#         metrics, train_valid = evaluate_Segmentation(model, nepochs, lossfn, lr, loaders, 
#                                                      device, ruta, verbose_train)
        
#         #Evaluate segmentation performance. Use the mean dice
#         dice=np.mean(metrics["dices"])
#     else:
#         dice = 0.9#?
    
#     # #Fitness as lienar combination of mean dice and the number of parameters
#     # fit = (1 - w)*dice + w*complexity
    
#     return 1-dice, params, #fit, dice, params#, metrics, train_valid, params
    
# def evaluationMP(ind, nepochs, lossfn, lr,
#                   max_params, w,
                 
#                   loaders, pset, device, ruta, verbose_train):
    
#     """Train, val and test loaders"""
#     train_loader, _ = loaders.get_train_loader() #loaders.get_train_loader(288, 480)
#     val_loader, _  = loaders.get_val_loader()
#     test_loader, _ = loaders.get_test_loader()
    

#     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     """Hyoperparameters for train"""
#     LOAD_MODEL = False
#     optimizer = optim.Adam(model.parameters(), lr = lr)
    
#     """Save model and images"""
#     save_model=False
#     save_images=False
    
#     """Train and valid model"""
#     train_loss, valid_loss, train_dice, valid_dice = train_and_validate(
#         model, train_loader, val_loader,
#         nepochs, optimizer, lossfn,
#         device, LOAD_MODEL, save_model,
#         ruta=ruta, verbose=verbose_train
#         )
    
#     """Test model"""
#     dices, ious, hds = test(test_loader, model, lossfn,
#                             save_imgs=save_images, ruta=ruta, device=device, verbose=verbose_train)
    
#     metrics={}
#     graphs={}
    
#     metrics["dices"]=dices
#     metrics["ious"]=ious
#     metrics["hds"]=hds
    
#     graphs["train_loss"]=train_loss
#     graphs["valid_loss"]=valid_loss
#     graphs["train_dice"]=train_dice
#     graphs["valid_dice"]=valid_dice
    
#     fitness = (1 - w)*np.mean(dices) + w*((max_params - params)/max_params)
#     print('Syntax tree:\t', str(ind), round(fitness,3), round(np.mean(dices),3), params)

#     #manager_list.append([num_process, fitness, np.mean(dices), params, train_loss, valid_loss, train_dice, valid_dice, dices, ious, hds])
#     # return fitness,
#     return fitness, np.mean(dices), params, metrics, graphs

"""Functions fot storage model, train and valid loss, graph as .png and .txt and segmented images"""
def save_ind(ind, ruta, filename='tree'):
    tree=stgp.PrimitiveTree(ind)
    nodes, edges, labels = stgp.graph(tree)#PrimitiveTree
    g = pgv.AGraph(directed=False)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]   
    
    txt_f=open(ruta + "/"+filename+".txt", "w")
    txt_f.write(str(ind))
    txt_f.write('Fitness:\t'+str(ind.fitness.values))
    txt_f.write('Dice:\t'+str(ind.dice))
    txt_f.write('Params:\t'+str(ind.params))
    txt_f.close()
    
    g.draw(ruta + "/" + filename + '.png')
    return

# """Shows a tree that represents an individual"""
def plt_ind(ind):
    tree=stgp.PrimitiveTree(ind)
    # print(dir(ind))
    nodes, edges, labels = stgp.graph(tree)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.axis('off')
    plt.show()
    return

"""Shows and save graph of valid an train loss"""
def save_graphtvd(ind, ruta, filename, show=False):
    train_loss=ind.train_dice
    valid_loss=ind.valid_dice
    epochs=[i for i in range(len(train_loss))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_loss, "b-", label="Train dice")
    line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid dice")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)

"""Shows and save graph of valid an train loss"""
def save_graphtv(ind, ruta, filename, show=False):
    train_loss=ind.train_loss
    valid_loss=ind.valid_loss
    epochs=[i for i in range(len(train_loss))]
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(epochs, train_loss, "b-", label="Train loss")
    line2 = ax1.plot(epochs, valid_loss, "r-", label="Valid loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    lns=line1+line2
    labs = [l.get_label() for l in lns]
    ax1.legend(line1+line2, labs, loc="center right")
    
    plt.close(fig)
    plt.show()
    
    #TrainValidationLoss.png
    fig.savefig(ruta+"/"+filename)

def identifier(ind, length=10):
    string=str(ind)
    return hashlib.sha224(string.encode('utf-8')).hexdigest()[-length:]
