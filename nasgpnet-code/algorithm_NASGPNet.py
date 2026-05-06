# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:06:05 2023

@author: josef
"""

from deap import tools
import pickle
import random
import numpy as np
from datetime import datetime, timedelta
from surrogate.metrics import pearson, coef_r2, kendall_tau, spearman, precisionK, mse
# from surrogate.utils import top_k, representative
from surrogate.selection import top_k, representative_k, complete
import matplotlib.pyplot as plt
# import time
import imageio
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# from surrogate import get_features

# from tqdm import tqdm
from utils.deap_utils import plot_surrogate
from utils.deap_utils import log2csv, show_statics, plot_nevals
import os
#%%Variation Operators
def MateMutation(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    mut_options = ['shrink','uniform','replace']#'replace'
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))  
            _parfitness = [ind1.fitness.values[0], ind2.fitness.values[0]]
            
            if hasattr(ind1, 'parfitness'):
                del ind1.parfitness
            # if hasattr(ind2, 'parfitness'):
            #     del ind2.parfitness
            # print("_parfitnessCX1", _parfitness)
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values, ind2.fitness.values
            ind1.parfitness = _parfitness
            # print("_parfitnessCX2", ind1.parfitness)
            # print('\t\t crossed\t', ind1)
            offspring.append(ind1)
            
        elif op_choice < cxpb + mutpb:  # Apply mutation
            mut_op = random.choice(mut_options)
            ind = toolbox.clone(random.choice(population))
            # print('indFit',ind.fitness.values)
            _parfitness = [ind.fitness.values[0]]
            if hasattr(ind, 'parfitness'):
                del ind.parfitness
            # print("_parfitnessMT1", _parfitness)
            # if mut_op=='ephem':
            #     ind, = toolbox.mutate_eph(ind)
            if mut_op=='shrink':
                ind, = toolbox.mutate_shrink(ind)
            elif mut_op=='replace':
                ind, = toolbox.mutate_replace(ind)
            elif mut_op=='uniform':
                ind, = toolbox.mutate_uniform(ind)
            del ind.fitness.values
            ind.parfitness = _parfitness
            # print("_parfitnessMT2", ind.parfitness)
            # print('\t\t mutation\t', ind)
            offspring.append(ind)
        else:                           # Apply reproduction?
            ind = toolbox.clone(random.choice(population))
            _parfitness = [ind.fitness.values[0]]
            if hasattr(ind, 'parfitness'):
                del ind.parfitness
            # print("_parfitnessMT1", _parfitness)
            ind, = toolbox.mutate_eph(ind)
            del ind.fitness.values
            ind.parfitness = _parfitness
            # print("_parfitnessMT1", _parfitness)
            
            offspring.append(ind)
            # print('\t\t mutated_eph \t', ind)
            # offspring.append(random.choice(population))

    return offspring

#%%Checkpoint
def checkpoint(generation, population, 
               # offspring,
               invalid_ind, 
               idx, elitism_inds,
               no_evs, delta_t, cache,
               halloffame, logbook,
               rndstate, ruta):
    
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(generation=generation, population=population, 
               # offspring=offspring,
               invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
               no_evs=no_evs, delta_t=delta_t, cache=cache, #archive=archive,
               halloffame=halloffame, logbook=logbook,
               rndstate=rndstate)   
    try: 
        with open(ruta + '/'+ "checkpoint_evo.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)
    except PermissionError:
        pass
    return


#%%Assign surrogates
def assign_attributes(ind, key, metrics, cache, toolbox, no_evs = 0, 
                      surrogate = None):
    if key in cache:
        ind.fitness.values = cache[key].fitness.values
        ind.params = cache[key].params
        for metric in metrics:
            setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
        # print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
        
        print('Syntax tree:\t', str(ind), ind.params)
        # print(round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
    else:
        if surrogate == None:
            fit, params, *out_metrics = toolbox.evaluate(ind)
            ind.fitness.values = fit,
            ind.params = params
            for metric, value in zip(metrics, out_metrics): #Segmentation metrics
                setattr(ind, metric._get_name(), value)
            print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")   
            # print(round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")
            
            cache[key] = ind
            no_evs += 1
        else:
            print('\t tree \t', ind)
            fit, params  = toolbox.evaluate_surrogate(ind, surrogate_model = surrogate) #!!toolbox surrogate
            ind.fitness.values = fit,
            ind.params = params   
            print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), ind.params, "\t in surrogate")
            # print(round(ind.fitness.values[0], 3), ind.params, "\t in surrogate")
     
    return ind, no_evs, cache

#%%baseline-algorithm
def baseline_eaNASGPNet(pop_size, toolbox, 
             # pset,
             cxpb, mutpb, ngen, nelit,
             ruta, checkpoint_name, 
             stats = None, halloffame=None, verbose_evo=__debug__):
    
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    metrics = toolbox.evaluate.keywords["metrics"]
    
    ####If checkpoint then load data from the file "/checkpoint"
    if checkpoint_name:
        print('recovering...', ruta)
        with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # print(cp.keys())
        start_gen    = cp["generation"]
        population   = cp["population"]
        invalid_ind  = cp["invalid_ind"]
        idx          = cp["idx"]
        elitism_inds = cp["elitism_inds"]
        no_evs       = cp["no_evs"]
        delta_t      = cp["delta_t"]
        cache        = cp["cache"]
        halloffame   = cp["halloffame"]
        logbook      = cp["logbook"]
        random.setstate(cp["rndstate"])
        
        print('Time:\t', delta_t)
        print('Start Get: \t', start_gen)
        print('Idx Pop: \t', idx)
        print('Best', str(tools.selBest(population, 1)[0]), 
              round(tools.selBest(population, 1)[0].fitness.values[0],3), 
              tools.selBest(population, 1)[0].params)
        
    else:
        ####Else start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'time', 'best', 
                          'best_DiceMetric',
                          'best_params'] + (stats.fields if stats else [])
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals
        idx=0
        no_evs=0
        cache={}
        
        # ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

    total_inds = len(invalid_ind)
    while len(invalid_ind)>0:#si no ha terminado, continua con evs
        
        ind = invalid_ind[0]#se toma el individuo de la primer posicion
        key = toolbox.identifier(ind)
        if key in cache:
            ind.fitness.values = cache[key].fitness.values
            ind.params = cache[key].params
            for metric in metrics:
                setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
            print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")

        else:
            fit, params, *out_metrics = toolbox.evaluate(ind)
            ind.fitness.values = fit,
            ind.params = params
            for metric, value in zip(metrics, out_metrics): #Segmentation metrics
                setattr(ind, metric._get_name(), value)
            print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")
            
            cache[key]=ind
            no_evs += 1
            
        ####Take time every evaluation
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t
        
        #update len of invalid ind
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        idx += 1
        
        #checkpoint
        checkpoint(generation=start_gen, 
                   population=population, 
                   invalid_ind=invalid_ind, 
                   idx=idx, elitism_inds=elitism_inds,
                   no_evs=no_evs, delta_t=delta_t, cache=cache,
                   halloffame=halloffame, logbook=logbook,
                   rndstate=random.getstate(), ruta=ruta)
        
        print(f"{idx}/{total_inds}", start_gen, ruta.split("/").pop(), delta_t)
        
    #best individual to store in logbook
    best_ind = tools.selBest(population, 1)[0]
    print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3),best_ind.params, start_gen)
    
    #append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    dict_log = {"gen":start_gen,
                "nevals":no_evs,
                "time":delta_t,
                "best":str(best_ind),
                "best_params":getattr(best_ind, "params"),}
    
    ##metrics of the best individual
    for m in metrics:
        key = str(type(m)).strip('>').strip("'").split('.')[-1]
        dict_log["best_"+key]=getattr(best_ind, key)
        
    logbook.record(**dict_log, **record)
    
    if verbose_evo:
        print(logbook.stream)
    
    #time
    #checkpoint
    #best_ind
    #log_record
    
    start_gen+=1
    for gen in range(start_gen, ngen+1):
        print('\nGen:\t', gen)
        
        #elitism
        elitism_inds = toolbox.selectElitism(toolbox.clone(population), k=nelit)

        #tournament
        offspring = toolbox.select(population, len(population) - nelit)
        
        #crossover and mutation
        offspring = MateMutation(offspring, toolbox, len(offspring), cxpb, mutpb) #4

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        idx = 0
        total_inds = len(invalid_ind)
        
        while len(invalid_ind)>0:
            ind = invalid_ind[0]
            key = toolbox.identifier(ind)
            if key in cache:
                ind.fitness.values = cache[key].fitness.values
                ind.params = cache[key].params
                for metric in metrics:
                    setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
            else:
                fit, params, *out_metrics = toolbox.evaluate(ind)
                ind.fitness.values = fit,
                ind.params = params
                for metric, value in zip(metrics,out_metrics):
                    setattr(ind, metric._get_name(), value)
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
                
                cache[key] = ind
                no_evs +=1
            
            #time
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            #update the population every evaluation in order to keep it in the checkpoint
            population = elitism_inds+offspring
            
            #update len of invalid ind
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            idx += 1
            
            #checkpoint
            checkpoint(generation=start_gen, 
                       population=population, 
                       invalid_ind=invalid_ind, 
                       idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)

            print(f"{idx}/{total_inds}", gen, ruta.split("/").pop(), delta_t)
            
        if halloffame is not None:
            halloffame.update(offspring)
            
        #time
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t #Keep delta_t, no matters loose t
        
        #checkpoint
        checkpoint(generation=start_gen, 
                   population=population, 
                   invalid_ind=invalid_ind, 
                   idx=idx, elitism_inds=elitism_inds,
                   no_evs=no_evs, delta_t=delta_t, cache=cache,
                   halloffame=halloffame, logbook=logbook,
                   rndstate=random.getstate(), ruta=ruta)
        #best ind
        best_ind = tools.selBest(population, 1)[0]
        print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3), best_ind.params, gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        dict_log = {"gen":start_gen,
                    "nevals":no_evs,
                    "time":delta_t,
                    "best":str(best_ind),
                    "best_params":getattr(best_ind, "params"),}
        
        for m in metrics:
            key = str(type(m)).strip('>').strip("'").split('.')[-1]
            dict_log["best_"+key]=getattr(best_ind, key)
        
        logbook.record(**dict_log,
                        **record)
        
        #For print
        if verbose_evo:
            print(logbook.stream)
    
    print("Time", delta_t)
        
    return population, logbook, cache



#%%Single generation experiment with multi-surrogates
def single_generation(pop_size, toolbox, pset,
                      surrogate_classes, surrogate_kwargs_list,
                      cxpb, mutpb, nelit, no_executions,
                      ruta,
                      ):
    surrogate_metrics = [coef_r2, pearson, spearman, kendall_tau]
    metrics = toolbox.evaluate.keywords["metrics"]
    
    # surr_names = [m.__name__ for m in surrogate_classes]
    # for i in range(4):
    #     surr_names[i] = str(surrogate_kwargs_list[i]['surrogate_model'])
        
    surr_names = ['RF', 'RF-14', 'KNN', 'SVM', 'GIN', 'GIN-R', 'TreeLSTM', 'TreeLSTM-R']
    
    # print('surrnames', surr_names)
    # print('metrics', surrogate_metrics)
    
    # Initialize logbook
    logbook = tools.Logbook()
    logbook.header = ['nevals', 'new_inds',# 'time', 
                      'best', 'best_dice', 'best_params'] + \
                        [f"{metric.__name__}-{i}" for i in surr_names for metric in surrogate_metrics] + \
                        [f"t_train-{i}" for i in surr_names]
    
    for exc in range(no_executions):
        cache = {}
        no_evs = 0
    
        #Generate the population
        population = toolbox.population(n=pop_size)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        #Evaluate the population usign the original fitness function
        for i, ind in enumerate(invalid_ind):
            key = toolbox.identifier(ind)
            ind, no_evs, cache = assign_attributes(ind, key, metrics, cache, toolbox, 
                                                   no_evs, surrogate=None)
            print(f"{i+1}/{len(invalid_ind)}")
            
        #Generate the archive from the initial population
        print('Len(cache)', len(cache))
        archive = list(cache.values())
        
        #Generate offspring- Single Generation Pass
        elitism_inds = toolbox.selectElitism(toolbox.clone(population), k=nelit)
        offspring = toolbox.select(population, len(population) - len(elitism_inds))
        offspring = MateMutation(offspring, toolbox, len(offspring), cxpb=cxpb, mutpb=mutpb)
        
        #update population
        population = elitism_inds + offspring
        
        # Xtest_ids = {toolbox.identifier(ind) for ind in population if toolbox.identifier(ind) not in cache}
        # print('len(Xtest)', len(Xtest_ids))
    
        #Initialize surrogate models                        
        surrogates = [cls(**kwargs) for cls, kwargs in zip(surrogate_classes, surrogate_kwargs_list)]
        surrogate_times = [0.0] * len(surrogates)
        
        #Train the surrogates
        for i, surrogate in enumerate(surrogates):
            print('Training surrogate\t', type(surrogate))
            Xtrain = archive
            ytrain = [ind.fitness.values[0] for ind in Xtrain]
            if type(surrogate) == "FeatureSurrogateModel":
                surrogate, time_surr = surrogate.fit(Xtrain, ytrain, first_gen = True, return_time = True)
            else:
                surrogate, time_surr = surrogate.fit(Xtrain, ytrain, return_time=True)
            surrogate_times[i] = time_surr#.total_seconds()
        
        # print('len(Xtest)',len(Xtest_ids))
        selected = [ind for ind in population if toolbox.identifier(ind) not in cache]
        selected = complete(selected) #la seleccion elimina los repetidos
        print('\tSelected', len(selected))
        
        #Real values on the selected individuals
        y_real=[]
        for ind in selected:
            print('Syntax tree:\t', str(ind))
            fit, params, *out_metrics = toolbox.evaluate(ind)
            y_real.append(fit)
            no_evs += 1
            #assign attributes in order to modigy the population variable and the best individuals of the first gen
            ind.fitness.values = fit,
            ind.params = params
            for metric, value in zip(metrics, out_metrics): #Segmentation metrics
                setattr(ind, metric._get_name(), value)
            print(round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")            
        y_real = np.array(y_real)
        
        #log entry
        best_ind = tools.selBest(population, 1)[0]
        log_entry = {"nevals":no_evs, "new_inds": len(selected),
                     "best": str(best_ind), 
                     "best_dice":best_ind.DiceMetric, 
                     "best_params":best_ind.params}
        
        for i, surrogate in enumerate(surrogates):
            y_pred = []
            for ind in selected:
                fit, _ = toolbox.evaluate_surrogate(ind, surrogate_model = surrogate)
                y_pred.append(fit)
            y_pred = np.array(y_pred)
            
            print('\ty_pred\t', y_pred)
            print('\ty_real\t', y_real)        
            
            # surr_metrics = {}
            for metric in surrogate_metrics:
                m = metric(y_real, y_pred)
                # surr_metrics[metric.__name__] = m
                log_entry[f"{metric.__name__}-{surr_names[i]}"] = m
                
                print(f"\t{metric.__name__}-{surr_names[i]}", m)
                
            log_entry[f"t_train-{surr_names[i]}"] = surrogate_times[i]
            print(f"\tt_train-{surr_names[i]}", surrogate_times[i])
            
            #recover metric to show in the plots
            textstr = '\n'.join((r'$R^2=%.3f$' % (log_entry["coef_r2"+f"-{surr_names[i]}"], ),
                                 r'$r=%.3f$' % (log_entry["pearson"+f"-{surr_names[i]}"]),
                                 r'$\rho=%.3f$' % (log_entry["spearman"+f"-{surr_names[i]}"]),
                                 r'$\tau=%.3f$' % (log_entry["kendall_tau"+f"-{surr_names[i]}"]),
                                 ))
            #also recover time
            textstr = '\n'.join((textstr, r'$t_{train}=%.3f$s' % (surrogate_times[i])))
            
            fig, ax = plt.subplots(figsize = (6, 6))
            sizes = np.random.uniform(15, 80, len(y_pred))
            colors = np.random.uniform(15, 80, len(y_real))
            
            ax.scatter(y_real, y_pred, s=sizes, c=colors, vmin=0, vmax=100)
            ax.plot([0, 1], [0, 1], color='black', linewidth = 0.2)
            ax.set_xlabel("real fit", fontsize = 14)
            ax.set_ylabel("predicted fit", fontsize = 14)
            ax.set_ylim([0, 1])
            ax.set_title(f"{surr_names[i]}")
            
            ax.tick_params(axis='x', labelsize = 14)
            ax.tick_params(axis='y', labelsize = 14)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.7, 0.5, textstr, transform=ax.transAxes, fontsize = 12,
                          verticalalignment='top', bbox = props)
            fig.tight_layout()
            title = f'-Exc-{exc+1}-Model-{surr_names[i]}-TrainSize-{len(archive)}-TestSize-{len(selected)}'
            plt.savefig(ruta+"/"+ruta.split("/").pop()+title+'.png', dpi=600)
            plt.close(fig)
            plt.show()
            
        #record the log_entry
        logbook.record(**log_entry)   
        print()
    
    return logbook


#%%Selection criteria comparison
def selection_criteria(pop_size, toolbox,
                       pset, 
                       surrogate_cls, 
                       surrogate_kwargs, # surrogate model and arguments
                       gen_updates, 
                       cxpb, mutpb, 
                       ngen, nelit,
                       ruta, checkpoint_name,
                       no_excecutions, k_selected,
                       stats = None, halloffame=None, 
                       verbose_evo = __debug__, 
                       save_images = False,
                       ):
    
    #metris to measrue prediction ability
    surrogate_metrics = [coef_r2, pearson, spearman, kendall_tau]
    
    #logbook de cada una de las 10 ejecuciones
    logbook = tools.Logbook()         
    logbook.header = ['nevals', 'new_inds', 'time',
                      'best', 'best_dice','best_params'] + \
                     [m.__name__ for m in surrogate_metrics] + ['t_train']
    

    for exc in range(no_excecutions):
        #path per execution
        path = ruta + f'/exc-{exc+1}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        #log de cada llamada
        pop, log, cache = baseline_surrogate(pop_size, toolbox, pset, 
                                             surrogate_cls, surrogate_kwargs, 
                                             gen_updates,
                                             cxpb, mutpb, ngen, nelit, 
                                             k_selected,
                                             path, checkpoint_name,
                                             stats=stats, halloffame=halloffame,
                                             verbose_evo=False,
                                             save_images = save_images)

        #save th log, pop, cache in a new path
        log2csv(log, stats, path)
        show_statics(log, path)
        plot_nevals(log, path) #evals along generations
        plot_surrogate(log, path)
        
        #recover important information about log to construct logbook
        log_entry = {"nevals":log.select("nevals")[-1], #np.mean([x for x in log.select("nevals") if x is not None]), 
                     "new_inds":np.mean([x for x in log.select("new_inds") if x is not None]),
                     "time":log.select("time")[-1],
                     "best":log.select("best")[-1],
                     "best_dice":log.select("best_DiceMetric")[-1], 
                     "best_params":log.select("best_params")[-1],
                     }
        
        # metrics = toolbox.evaluate.keywords["metrics"]
        # for m in metrics:
        #     key = str(type(m)).strip('>').strip("'").split('.')[-1]
        #     log_entry["best_"+key] = getattr(best_ind, key)
            
        
        for m in surrogate_metrics:
            log_entry[m.__name__] = np.mean([x for x in log.select(m.__name__) if x is not None])
        
        log_entry["t_train"]=np.mean([x for x in log.select("t_train") if x is not None])
        logbook.record(**log_entry)
        
    log2csv(logbook, stats, ruta)
        
    return pop, log, cache

#%%baseline surrogate
def baseline_surrogate(pop_size, toolbox,
                       pset, 
                       surrogate_cls, surrogate_kwargs, # surrogate model and arguments
                       gen_updates, 
                       cxpb, mutpb, ngen, nelit,
                       k,
                       ruta, checkpoint_name,
                       stats = None, halloffame=None, 
                       verbose_evo = __debug__, 
                       save_images = True):
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    metrics = toolbox.evaluate.keywords["metrics"]
    surrogate_metrics = [pearson, coef_r2, 
                         kendall_tau, spearman]
                         # precisionK]
    axes2gif = []
    
    ####If checkpoint then load data from the file "/checkpoint"
    if checkpoint_name:
        print('recovering...', ruta)
        with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # print(cp.keys())
        start_gen    = cp["generation"]
        population   = cp["population"]
        invalid_ind  = cp["invalid_ind"]
        idx          = cp["idx"]
        elitism_inds = cp["elitism_inds"]
        no_evs       = cp["no_evs"]
        delta_t      = cp["delta_t"]
        cache        = cp["cache"]
        halloffame   = cp["halloffame"]
        logbook      = cp["logbook"]
        random.setstate(cp["rndstate"])
        
        print('Time:\t', delta_t)
        print('Start Get: \t', start_gen)
        print('Idx Pop: \t', idx)
        print('Best', str(tools.selBest(population, 1)[0]), 
              round(tools.selBest(population, 1)[0].fitness.values[0],3), 
              tools.selBest(population, 1)[0].params)
        
    else:
        ####Else start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        
        logbook.header = ['gen', 'nevals', 'new_inds', 
                          'time', 'best', 
                          'best_DiceMetric', 'best_params'
                          ] + [m.__name__ for m in surrogate_metrics] + ['t_train'] + (stats.fields if stats else [])
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals (r the same?)
        idx = 0
        no_evs = 0
        cache = {}
        
        # ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
        #initial surrogate is not activate
        surrogate_model = None
    
    total_inds = len(invalid_ind)
    
    while len(invalid_ind)>0:
        ind = invalid_ind[0] #take the first individual
        key = toolbox.identifier(ind)
        
        # print("Checking srting\t:", str(ind))
        ind, no_evs, cache = assign_attributes(ind, key, metrics, cache, toolbox, no_evs, 
                                               surrogate = surrogate_model)
        
        ####Take time every evaluation
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t
        
        #update len of invalid ind
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        idx += 1
        
        #checkpoint
        checkpoint(generation=start_gen, 
                   population=population, 
                   invalid_ind=invalid_ind, 
                   idx=idx, elitism_inds=elitism_inds,
                   no_evs=no_evs, delta_t=delta_t, cache=cache,
                   halloffame=halloffame, logbook=logbook,
                   rndstate=random.getstate(), ruta=ruta)
        
        print(f"{idx}/{total_inds}", start_gen, ruta.split("/").pop(), delta_t)
    
    # update archive with non_repeted individuals, taken from the cache
    print('Len(cache)', len(cache))
    archive = list(cache.values())

    #best individual to store in logbook
    best_ind = tools.selBest(population, 1)[0]
    print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3),best_ind.params, start_gen)
    
    #append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    dict_log = {"gen":start_gen,
                "nevals":no_evs,
                "time":delta_t,
                "best":str(best_ind),
                "best_params":getattr(best_ind, "params"),
                }
    
    ##metrics of the best individual
    for m in metrics:
        key = str(type(m)).strip('>').strip("'").split('.')[-1]
        dict_log["best_"+key]=getattr(best_ind, key)
        
    logbook.record(**dict_log, **record)
    
    if verbose_evo:
        print(logbook.stream)
        
    start_gen += 1
    for gen in range(start_gen, ngen+1):
        print('\nGen:\t', gen)
        
        #elitism
        elitism_inds = toolbox.selectElitism(toolbox.clone(population), k=nelit)
        
        #tournament
        offspring = toolbox.select(population, len(population) - nelit)
        
        #crossover and mutation
        offspring = MateMutation(offspring, toolbox, len(offspring), cxpb, mutpb) #4
        
        #evaluate individuals with invalid fitness using real and surrogate_model
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        #init counter and the total of individuals to evaluate
        idx = 0
        total_inds = len(invalid_ind)
        
        #select non-repeated elements that dont appear in cache (before change cache with assign_attribute function)
        Xtest_ids = [toolbox.identifier(ind) for ind in offspring if toolbox.identifier(ind) not in cache]
        Xtest_ids = set(Xtest_ids)
        # Xtest_ids = {toolbox.identifier(ind) for ind in offspring if toolbox.identifier(ind) not in cache}
        #que no esten en cache
        
        # _total =  len(cache) + len(Xtest_ids)
        # datasplit_test_ratio =  len(Xtest_ids)/_total
        # print('SplitData', _total, datasplit_test_ratio)
        # print(datasplit_test_ratio > 0.1)
        
        #reconstructing the surrogate model
        _model_reconstruction = gen==1 or (gen>1) and (gen-1)%gen_updates==0 #datasplit_test_ratio > 0.1
        if _model_reconstruction: # it could be another condition
            print('Constructing surrogate model')
            #build the surrogate model
            Xtrain = archive
            ytrain = [ind.fitness.values[0] for ind in Xtrain] #fitness of individuals
            surrogate_model = surrogate_cls(**surrogate_kwargs)
            surrogate_model, time_surr = surrogate_model.fit(Xtrain, ytrain, first_gen=gen == 1, return_time=True)#si es la generacion uno nan se sustituye con ceros
        
        #evaluate each individual in the invalid_ind list
        #using the real fitness function and the surrogate model
        while len(invalid_ind)>0:# or len(ytest)>0:
            ind = invalid_ind.pop(0) #update len of invalid_ind
            key = toolbox.identifier(ind)
            
            # print("Checking srting\t:", str(ind))
            
            #evaluate using the cache or the surrogate model
            ind, no_evs, cache = assign_attributes(ind, key, metrics, cache, toolbox, no_evs, 
                                                   surrogate = surrogate_model)
                
            #time
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            #update the population every evaluation in order to keep it in the checkpoint
            population = elitism_inds + offspring
            
            #update len of invalid ind
            #invalid_ind = #[ind for ind in population if not ind.fitness.valid]
            idx += 1
            
            #checkpoint
            checkpoint(generation=gen, 
                       population=population, 
                       invalid_ind=invalid_ind, 
                       idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)

            print(f"{idx}/{total_inds}", gen, ruta.split("/").pop(), delta_t)
        
        
        print('Len(Xtest):', len(Xtest_ids), Xtest_ids)
        #Representative selection to train in the original fitness function (p-top) and them to the cache
        if _model_reconstruction:
            print('\tTraining in real fitness function')
            # print('Len(Xtest):', len(Xtest_ids), Xtest_ids)
            
            ytest = [] #data to test the surrogate model
            y_hat = []
            
            #select the never-seen-before individuals
            selected = [ind for ind in population if toolbox.identifier(ind) in Xtest_ids]
            
            #representative selection
            if len(selected)>0:
                selected = representative_k(selected, k=k)
                # selected = complete(selected)
                print('Selected', len(selected))
                # print('FitnessSelected:\t', [ind.fitness.values for ind in selected])
            
                #eval the selected inviduals in the real fitneess function            
                for ind in selected:
                    key = toolbox.identifier(ind)
                    if key not in cache:##??? n se supone que estos NO estan en la cache?
                        real, no_evs, cache = assign_attributes(toolbox.clone(ind), key, metrics, cache, toolbox, no_evs, 
                                                                surrogate = None) # Real fitness in order to update the cache
                        ytest.append(real.fitness.values[0])
                        y_hat.append(ind.fitness.values[0])
                        ind.fitness.values = real.fitness.values #Replace the predicted fitness of the individual by the real fitness ##!!!
             
                #to match with the metrics
                ytest = np.array(ytest)
                y_hat = np.array(y_hat)
                
                # print('FitnessSelected2:\t', [ind.fitness.values for ind in selected])
                print('Len(ytest)', len(ytest), ytest)
                print('Len(y_hat)', len(y_hat), y_hat)
                print('Len(cache)', len(cache))
                print('NoEvs', no_evs)
            
                #Evaluate the surrogate model with the selected indidivuals
                surr_metrics = {}
            
                #evaluate surrogate_model
                # if len(ytest)>1:#/len(cache) > datasplit_ratio_th:
                surr_metrics = {func.__name__: func(ytest, y_hat) for func in surrogate_metrics}
                print('Surrogate Metrics', surr_metrics)
                
                #Generate plot for the surrogate model predictions
                if save_images:
                    fig, ax = plt.subplots(figsize = (6, 6))
                    sizes = np.random.uniform(15, 80, len(ytest))
                    colors = np.random.uniform(15, 80, len(ytest))
                    
                    ax.scatter(ytest, y_hat, s=sizes, c=colors, vmin=0, vmax=100)
                    ax.plot([0, 1], [0, 1], color='black', linewidth = 0.2)
                    ax.set_xlabel("real fit", fontsize = 14)
                    ax.set_ylabel("predicted fit", fontsize = 14)
                    ax.set_ylim([0, 1])
                    ax.set_title(surrogate_kwargs["surrogate_model"].__class__.__name__+"-Gen:"+str(gen))
                    ax.tick_params(axis='x', labelsize = 14)
                    ax.tick_params(axis='y', labelsize = 14)
                    textstr = '\n'.join((r'$R^2=%.3f$' % (surr_metrics["coef_r2"], ),
                                         r'$r=%.3f$' % (surr_metrics["pearson"]),
                                         r'$\rho=%.3f$' % (surr_metrics["spearman"]),
                                         r'$\tau=%.3f$' % (surr_metrics["kendall_tau"]),
                                         # r'$P@K=%.3f$' % (surr_metrics["precisionK"]),
                                         # r'$mse=%.3f$' % (surr_metrics["mse"]),
                                         ))
                    textstr = '\n'.join((textstr, r'$t_{train}=%.3f$s' % (time_surr))) if _model_reconstruction else textstr
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.7, 0.5, textstr, transform=ax.transAxes, fontsize = 12,
                                  verticalalignment='top', bbox = props)
                    fig.tight_layout()
                    title = f'-Gen-{gen}-TrainSize-{len(archive)}-TestSize-{len(ytest)}'
                    plt.savefig(ruta+"/"+ruta.split("/").pop()+title+'.png', dpi=600)
                    plt.close(fig)
                    plt.show()
                    
                    axes2gif.append(fig)
    
            #update archive
            # archive = archive + Xtest
            archive = list(cache.values())
            print('\n')
        
        if halloffame is not None:
            halloffame.update(offspring)
            
        #time
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t #Keep delta_t, no matters loose t
        
        #checkpoint
        checkpoint(generation=gen, 
                   population=population, 
                   invalid_ind=invalid_ind, 
                   idx=idx, elitism_inds=elitism_inds,
                   no_evs=no_evs, delta_t=delta_t, cache=cache,
                   halloffame=halloffame, logbook=logbook,
                   rndstate=random.getstate(), ruta=ruta)
        
        #best ind
        best_ind = tools.selBest(population, 1)[0]
        print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3), best_ind.params, gen)
        
        print('Len(Xtest)2:', len(Xtest_ids), Xtest_ids)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        dict_log = {"gen":gen,
                    "nevals":no_evs,
                    "new_inds":len(Xtest_ids),
                    "time":delta_t,
                    "best":str(best_ind),
                    "best_params":getattr(best_ind, "params")}
        
        dict_log = {**dict_log, **({"t_train": time_surr} if _model_reconstruction else {})}
        dict_log = {**dict_log, **(surr_metrics if _model_reconstruction else {})} #uptade dict log with surrogate metrics 
        
        for m in metrics:
            key = key = str(type(m)).strip('>').strip("'").split('.')[-1]
            dict_log["best_"+key]=getattr(best_ind, key)
        
        logbook.record(**dict_log,
                        **record)
        
        #For print
        if verbose_evo:
            print(logbook.stream)
            
    #generate a GIF
    if save_images:
        frames = []
        for fig in axes2gif:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            # frames.append(np.array(fig.canvas.renderer._renderer))
        title = f'-Gen-{gen}'
        imageio.mimsave(ruta+"/"+ruta.split("/").pop()+title+'.gif', frames, duration=0.5)
    print("Time", delta_t)    
    
    return population, logbook, cache
      

#%%NASGP-Net Algorithm
# def eaNASGPNet(pop_size, toolbox, cxpb, mutpb, ngen, nelit,
#                  ruta, checkpoint_name,
#                  stats=None, halloffame=None, verbose_evo=__debug__,
#                  ):
    
#     ####Take time. Keep delta_t, no matters loose t
#     init_time=datetime.now()
#     delta_t=timedelta(seconds=0)
#     metrics = toolbox.evaluate.keywords["metrics"]
    
#     ####If checkpoint then load data from the file "/checkpoint"
#     if checkpoint_name:
#         print('recovering...', ruta)
#         with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
#             cp = pickle.load(cp_file)
#         # print(cp.keys())
#         start_gen    = cp["generation"]
#         population   = cp["population"]
#         invalid_ind  = cp["invalid_ind"]
#         idx          = cp["idx"]
#         elitism_inds = cp["elitism_inds"]
#         no_evs       = cp["no_evs"]
#         delta_t      = cp["delta_t"]
#         cache        = cp["cache"]
#         halloffame   = cp["halloffame"]
#         logbook      = cp["logbook"]
#         random.setstate(cp["rndstate"])
        
#         print('Time:\t', delta_t)
#         print('Start Get: \t', start_gen)
#         print('Idx Pop: \t', idx)
#         print('Best', str(tools.selBest(population, 1)[0]), 
#               round(tools.selBest(population, 1)[0].fitness.values[0],3), 
#               tools.selBest(population, 1)[0].params)
        
#     else:
#         ####Else start a new evolution
#         # random.seed(42)#!!!
#         population = toolbox.population(n=pop_size)
#         start_gen  = 0
#         halloffame = tools.HallOfFame(maxsize=nelit)
#         logbook = tools.Logbook()
#         logbook.header = ['gen', 'nevals', 'time', 'best', 
#                           'best_dice',
#                           'best_iou',
#                           'best_hd',
#                           'best_hd95',
#                           'best_nsd',
#                           'best_params'] + (stats.fields if stats else [])
#         offspring = []
#         elitism_inds = []
        
#         ###Count the number of evaluations and evaluated individuals
#         idx=0
#         no_evs=0
#         cache={}
        
#         # ###Individuals to evaluate
#         invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
#     #%%%%Start option 1
#     #Si no ha terminado con invalid ind y sigue en la generacion 0
#     ###Individuals to evaluate
#     # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
#     if idx<len(invalid_ind): #and start_gen == 0:
#         while idx < len(invalid_ind):
#             ind = invalid_ind[idx]
#             key = toolbox.identifier(ind)
            
#             #Predict using original fitness function
#             # ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
            
#             if key in cache:
#                 #Assign attributes from cache
#                 ind.fitness.values = cache[key].fitness.values
#                 ind.params = cache[key].params

#                 for metric in metrics: #Segmentation metrics
#                     setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
     
#                 print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
                
#             else:
                
#                 if start_gen==0 and idx==0:
#                     #Esto de guardarlos datos, para qué era? llevar un seguimiento 
#                     #de los datos de entrenamiento y prueba por fold
#                     fit, params, *out_metrics = toolbox.evaluate(ind, save_data=True)
#                 else:
#                     fit, params, *out_metrics  = toolbox.evaluate(ind)
                
#                 ind.fitness.values = fit,
#                 ind.params = params
#                 for metric, value in zip(metrics, out_metrics): #Segmentation metrics
#                     setattr(ind, metric._get_name(), value)
                
#                 #Add to cache
#                 cache[key]=ind
                
#                 ####Increment the number of evaluations when original objective function is used
#                 ####and key is not in cache
#                 no_evs+=1
                
#                 print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")

#             ####Increment the number of evaluated individuals from invalid ind
#             idx+=1
            
#             ####Take time every evaluation
#             t = datetime.now()
#             delta_t += (t - init_time)
#             init_time = t #Keep delta_t, no matters loose t
            
#             ####Checkpoint every evaluation and every generation
#             checkpoint(generation=start_gen, 
#                        population=population, 
#                        invalid_ind=invalid_ind, 
#                        idx=idx, elitism_inds=elitism_inds,
#                        no_evs=no_evs, delta_t=delta_t, cache=cache,
#                        halloffame=halloffame, logbook=logbook,
#                        rndstate=random.getstate(), ruta=ruta)
            
#             print(f"{idx}/{len(invalid_ind)}", start_gen, ruta.split("/").pop(), delta_t)
         
#         best_ind = tools.selBest(population, 1)[0]
#         print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3),best_ind.params, start_gen)
        
#         # Append the current generation statistics to the logbook
#         record = stats.compile(population) if stats else {}
#         dict_log = {"gen":start_gen,
#                     "nevals":no_evs,
#                     "time":delta_t,
#                     "best":str(best_ind)}
#         for m in metrics:
#             key = str(type(m)).strip('>').strip("'").split('.')[-1]
#             dict_log["best_"+key]=getattr(best_ind, key)
            
#         logbook.record(**dict_log,
#                        **record)
        
#         #For print
#         if verbose_evo:
#             print(logbook.stream)
        
    
#     #%%%Start option 2
#     #Si ya ha terminado con "invalid_ind" en la sesión anterior o si no checkpoint_name, 
#     #entonces continúa normalmente (pasa a la siguiente generación)
#     # print("Continue Here\n\n")
#     if idx==len(invalid_ind) or checkpoint_name==False:
#         start_gen+=1
        
#         for gen in range(start_gen, ngen+1):
#             print('\nGen:\t', gen)
            
#             #Elitism
#             population_for_eli=[toolbox.clone(ind) for ind in population]
#             elitism_inds = toolbox.selectElitism(population_for_eli, k=nelit)
            
#             #Tournament selection
#             offspring = toolbox.select(population, len(population) - nelit)
            
#             #Crossover and Mutation
#             offspring = MateMutation(offspring, toolbox, len(offspring), cxpb, mutpb)
            
#             invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#             idx = 0
            
#             #Evaluate using cache or original model;
#             while idx < len(invalid_ind):
#                 ind = invalid_ind[idx]
#                 key = toolbox.identifier(ind)
                
#                 # ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
#                 if key in cache:
#                     #Assign attributes from cache
#                     ind.fitness.values = cache[key].fitness.values
#                     ind.params = cache[key].params
                    
#                     for metric in metrics: #Segmentation metrics
#                         setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
                            
#                     print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
                    
#                 else:
#                     # #Assign attributes from the original objective function                  
#                     fit, params, *out_metrics  = toolbox.evaluate(ind)
                
#                     ind.fitness.values = fit,
#                     ind.params = params
#                     for metric, value in zip(toolbox.evaluate.keywords["metrics"], out_metrics): #Segmentation metrics
#                         setattr(ind, metric._get_name(), value)
                    
#                     #Add to cache
#                     cache[key]=ind
                    
#                     ####Increment the number of evaluations when original objective function is used
#                     ####and key is not in cache
#                     no_evs+=1
                    
#                     print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")
                
#                 ####Increment the number of evaluated individuals from invalid ind
#                 idx+=1
                
#                 ####Take time every evaluation
#                 t = datetime.now()
#                 delta_t += (t - init_time)
#                 init_time = t #Keep delta_t, no matters loose t
                
#                 population = elitism_inds+offspring
                
#                 ####Checkpoint every evaluation and every generation
#                 checkpoint(generation=gen, 
#                            population=population, 
#                            # offspring=offspring,
#                            invalid_ind=invalid_ind, 
#                            idx=idx, elitism_inds=elitism_inds,
#                            no_evs=no_evs, delta_t=delta_t, cache=cache,
#                            halloffame=halloffame, logbook=logbook,
#                            rndstate=random.getstate(), ruta=ruta)
                
#                 print(f"{idx}/{len(invalid_ind)}", gen, ruta.split("/").pop(), delta_t)
                
#                 # #Save test, train, split list
#                 # if idx == len(invalid_ind) and gen == ngen:
                    
                
#             #Back the elitism individuals to population
#             # offspring.extend(elitism_inds)
            
#             # Update the hall of fame with the generated individuals
#             if halloffame is not None:
#                 halloffame.update(offspring)
            
#             #Replace the curren population by the offspring
#             # population[:] = offspring
            
#             ####Take time every evaluation
#             t = datetime.now()
#             delta_t += (t - init_time)
#             init_time = t #Keep delta_t, no matters loose t
            
#             ####Checkpoint every evaluation and every generation
#             checkpoint(generation=gen, population=population, 
#                        # offspring=offspring,
#                        invalid_ind=invalid_ind, 
#                        idx=idx, elitism_inds=elitism_inds,
#                        no_evs=no_evs, delta_t=delta_t, cache=cache,
#                        halloffame=halloffame, logbook=logbook,
#                        rndstate=random.getstate(), ruta=ruta)
            
#             #Take the best individual after finishes each generation
#             #and print their attributes
#             best_ind = tools.selBest(population, 1)[0]
#             # print(str(best_ind), best_ind.fitness.values, best_ind.dice, best_ind.params)
#             print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3), best_ind.params, gen)
            
#             # Append the current generation statistics to the logbook
#             record = stats.compile(population) if stats else {}
#             dict_log = {"gen":start_gen,
#                         "nevals":no_evs,
#                         "time":delta_t,
#                         "best":str(best_ind)}
#             for m in metrics:
#                 key = key = str(type(m)).strip('>').strip("'").split('.')[-1]
#                 dict_log["best_"+key]=getattr(best_ind, key)
            
#             logbook.record(**dict_log,
#                             **record)
            
#             #For print
#             if verbose_evo:
#                 print(logbook.stream)
            
#         print("Time", delta_t)
        
#         return population, logbook, cache# archive, cache