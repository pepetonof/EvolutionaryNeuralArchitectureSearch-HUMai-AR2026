# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 06:54:23 2021

@author: josef
"""
import numpy as np
import random
# from deap import tools
# import matplotlib.pyplot as plt
import pickle
import os
from torchinfo import summary

#%%Save progress: CHECKPOINT
def save_progress(ruta, filename,
                  population, best, 
                  gen,
                  halloffame, logbook,
                  
                  offspring,
                  elitism_inds,
                  invalid_ind, idx, 
                  no_evs, delta_t,
                  cache
                  ):
    
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(population=population, best=best, generation=gen,
              halloffame=halloffame, logbook=logbook,
              
              offspring=offspring,
              elitism_inds=elitism_inds,
              invalid_ind=invalid_ind, idx=idx, 
              no_evs=no_evs, delta_t=delta_t,
              cache=cache,
              rndstate=random.getstate())
    with open(ruta + '/'+ filename, "wb") as cp_file:
        pickle.dump(cp, cp_file)
    del cp_file
    
    return

#%%Save infromation of entire execution

def save_execution(ruta, filename, pop, log, cache, best_model, archive=None): #archive,
    cp = dict(pop=pop, log=log, 
              cache=cache,
              best_model=best_model)
    
    if archive!=None:
        cp["archive"]=archive
    
    with open(ruta + '/'+ filename, "wb") as cp_file:
        pickle.dump(cp, cp_file)
    del cp_file
    
    return
    
#%%Save results
def saveResults(filename, *args, **kargs):
    f=open(filename, 'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return

#%%Save evolutionary Details
def saveEvoDetails(population, elitism,
                   crossr, mutr, tournsize,
                   ngen, evoTime,
                   best, bestfitness,
                   dices, ious, hds,
                   filename,
                ):
    saveResults(filename,
                'Population:', population, 'Elitism:', elitism,
                'CrossRate:', crossr,
                'MutRate:', mutr,
                'TournSize:', tournsize,
                'NGen:', ngen, 'EvoTime:', evoTime,
                'Best:', best, 'Best Fitness', bestfitness,
                
                'DiceMean:', np.mean(dices), 'DiceMax:', np.max(dices),
                'DiceMin:', np.min(dices), 'DiceStd:', np.std(dices),
                
                'IoUMean:', np.mean(ious), 'IoUMax:', np.max(ious),
                'IoUMin:', np.min(ious), 'IoUStd:', np.std(ious),
                
                'HdMean:', np.mean(hds), 'HdMax:', np.max(hds),
                'HdMin:', np.min(hds), 'HdStd:', np.std(hds),
                )
    return

def saveEvolutionaryDetails(evolutionary_params, best, no_evs, time, filename):
    d={}
    d.update(evolutionary_params)
    d_aux={"Best":best, "Best_Fitness":best.fitness.values[0], 
           
           "DiceMean":best.dice, "Params":best.params,
           
           # "DiceMean": np.mean(best.dices), "DiceMax":np.max(best.dices),
           # "DiceMin": np.min(best.dices), "DiceStd":np.std(best.dices),
           
           # "IoUMean": np.mean(best.ious), "IoUMax": np.max(best.ious),
           # "IoUMin": np.min(best.ious), "IoUStd": np.std(best.ious),
           
           # "HdMean": np.mean(best.hds), "HdMax": np.max(best.hds),
           # "HdMin": np.min(best.hds), "HdStd": np.std(best.hds),
           }
    d.update(d_aux)
    d_aux={"No_Evs": no_evs, "Time": time}
    d.update(d_aux)
    
    with open(filename, 'w') as f: 
        for key, value in d.items():
            # f.write('%s\n', key)
            # f.write('%s\n', value)
            f.write('%s:%s\n' % (key, value))
    return
    
    
#%%Save Training Details
def saveTrainDetails(train_size, valid_size,
                      learning_rate, nepochs,
                      im_h, im_w,
                      loss_fn, optimizer,                      
                      dices, ious, hds,
                      # dices, dice_std,
                      # dice_min, dice_max,
                      summary_model,
                      filename,
                      ):
    saveResults(filename, 
                'Train_size:', train_size, 'Valid_size:', valid_size,
                'Learning_rate:', learning_rate, 'Num_epochs:', nepochs,
                'Image_height:', im_h, 'Image_width:', im_w,
                'Loss_fn:', loss_fn, 'Optimizer:', optimizer,
                
                'DiceMean:', np.mean(dices), 'DiceMax:', np.max(dices),
                'DiceMin:', np.min(dices), 'DiceStd:', np.std(dices),
                
                'IoUMean:', np.mean(ious), 'IoUMax:', np.max(ious),
                'IoUMin:', np.min(ious), 'IoUStd:', np.std(ious),
                
                'HdMean:', np.mean(hds), 'HdMax:', np.max(hds),
                'HdMin:', np.min(hds), 'HdStd:', np.std(hds),
                
                'Summary_Model:', summary_model,
                )
    return

# def saveTrainingDetails(training_parameters, loaders, 
#                         best, summary_model, filename):
#     d={}
#     d.update(training_parameters)
    
#     d_aux={"Height": loaders.IMAGE_HEIGHT, "Width":loaders.IMAGE_WIDTH, 
#            "Train_Size":len(loaders.TRAIN_IMG_DIR),
#            "Valid_Size":len(loaders.VAL_IMG_DIR),
#            "Test_Size":len(loaders.TEST_IMG_DIR),
#            }
#     d.update(d_aux)
    
#     # d_aux={"Best":str(best), "Best_Fitness":best.fitness.values[0], 
           
#     #        "DiceMean":best.dice, "Params":best.params,
#     #        }
#     # d.update(d_aux)
    
#     d_aux={"Best":best, "Best_Fitness":best.fitness.values[0], 
           
#             "DiceMean": np.mean(best.dices), "DiceMax":np.max(best.dices),
#             "DiceMin": np.min(best.dices), "DiceStd":np.std(best.dices),
           
#             "IoUMean": np.mean(best.ious), "IoUMax": np.max(best.ious),
#             "IoUMin": np.min(best.ious), "IoUStd": np.std(best.ious),
           
#             "HdMean": np.mean(best.hds), "HdMax": np.max(best.hds),
#             "HdMin": np.min(best.hds), "HdStd": np.std(best.hds),
#             }
#     d.update(d_aux)
    
#     d_aux={"Summary_Model": summary_model}
#     d.update(d_aux)
    
#     with open(filename, 'w', encoding="utf-8") as f: 
#         for key, value in d.items(): 
#             f.write('%s:%s\n' % (key, value))
    
#     return
    

def saveTrainingDetails(ruta, model, metricsTest, metricsVal, lossAndDice, 
                        train_set, valid_set, test_set, dloaders, 
                        batch_size, in_channels, image_height, image_width,
                        ):
    path_data = f"{ruta}/data/"
    
    #Save data of metrics in as .pkl file
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    with open(f"{path_data}/metricsTest.pkl", "wb") as cp_file:
        pickle.dump(metricsTest, cp_file)
    with open(f"{path_data}/metricsVal.pkl", "wb") as cp_file:
        pickle.dump(metricsVal, cp_file)
    with open(f"{path_data}/lossAndDice.pkl", "wb") as cp_file:
        pickle.dump(lossAndDice, cp_file)
    
    #Save data as .txt with statistics
    d={}
    d_aux = {"Height": dloaders.IMAGE_HEIGHT, 
             "Width":dloaders.IMAGE_WIDTH, 
             "Train_Size":len(train_set["images"]),#len(dloaders.TRAIN_IMG_DIR),
             "Valid_Size":len(train_set["images"]),#len(dloaders.VAL_IMG_DIR),
             "Test_Size": len(train_set["images"])#len(dloaders.TEST_IMG_DIR),
             # "Ind": str(ind), 
             # "Ind_Fitness":(1 - w)*np.mean(metricsTest["DiceMetric"]) + w*complexity
             }
    d.update(d_aux)
    
    d_aux={}
    statics = [np.mean, np.median, np.max, np.min, np.std]
    for key, value in metricsTest.items():
        for s in statics:
            composed_key=key+'_'+s.__name__
            d_aux[composed_key]=s(value)
    d.update(d_aux)

    model_stats=summary(model, (batch_size, in_channels, image_height, image_width), verbose=0)
    summary_model = str(model_stats)
    d_aux = {"Summary_Model": summary_model}
    d.update(d_aux)

    with open(f"{path_data}/data.txt", 'w', encoding="utf-8") as f: 
        for key, value in d.items(): 
            f.write('%s\n%s\n' % (key, value))
            
    #Save the train_set, valid_set and test_set in a .txt files
    with open(f"{path_data}/train_set.txt", 'w', encoding="utf-8") as f: 
        for key, value in train_set.items(): 
            f.write('\n%s\n' % (key))
            for name in [path.name for path in value]:
                f.write('%s\n' % (name))
                        
    with open(f"{path_data}/valid_set.txt", 'w', encoding="utf-8") as f: 
        for key, value in valid_set.items(): 
            f.write('\n%s\n' % (key))
            for name in [path.name for path in value]:
                f.write('%s\n' % (name))
                        
    with open(f"{path_data}/test_set.txt", 'w', encoding="utf-8") as f: 
        for key, value in test_set.items(): 
            f.write('\n%s\n' % (key))
            for name in [path.name for path in value]:
                f.write('%s\n' % (name))