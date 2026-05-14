import sys
sys.path.append('C:/Users/josef/OneDrive/Documentos/DIA/Talleres/Taller-HumaiAR2026/EvolutionaryNeuralArchitectureSearch-HUMai-AR2026/nasgpnet-code')

#%% Import
import os
from functools import partial
from deap import tools
from deap import creator, base
import operator
import stgp
from strongGPDataType import (moduleTorch, moduleTorchL, moduleTorchSe, moduleTorchCn, moduleTorchCt, moduleTorchP,
                              outChConv, outChSConv, kernelSizeConv, dilationRate,
                              tetha,wArithm)
from operators.functionSet import (convolution, sep_convolution,
                             res_connection, dense_connection,
                             se,
                             add, sub, cat,
                             maxpool, avgpool)

from toolbox_functions import make_model, identifier, plt_ind
from data.dataset import Dataset2D
import data.dataSplit as dataSplit
from metrics.segmentation_metrics import DiceMetric
from losses.loss_functions import DiceLoss

#%% Pset
pset = stgp.PrimitiveSetTyped("main", [moduleTorch], moduleTorchP)

#Number of blocks layer??

#Pooling Layer
pset.addPrimitive(maxpool, [moduleTorchL], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchL], 
                  moduleTorchP, name='apool')

pset.addPrimitive(maxpool, [moduleTorchCn], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchCn], 
                  moduleTorchP, name='apool')

pset.addPrimitive(maxpool, [moduleTorchSe], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchSe], 
                  moduleTorchP, name='apool')

pset.addPrimitive(maxpool, [moduleTorchCt], 
                  moduleTorchP, name='mpool')
pset.addPrimitive(avgpool, [moduleTorchCt], 
                  moduleTorchP, name='apool')


#Feature Construction Layer Optional
#L,L; L,Cn; Cn,L; Cn,Cn; 
pset.addPrimitive(add, [moduleTorchL, wArithm, moduleTorchL, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchL, wArithm, moduleTorchL, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchL, moduleTorchL], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchL, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchL, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchL, moduleTorchCn], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchCn, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchCn, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchCn, moduleTorchL], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchCn, wArithm, moduleTorchCn, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchCn, wArithm, moduleTorchCn, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchCn, moduleTorchCn], 
                  moduleTorchCt, name='cat')

# Se,Se; Se,L; L,Se;
pset.addPrimitive(add, [moduleTorchSe, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchSe, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchSe, moduleTorchSe], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchSe, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchSe, wArithm, moduleTorchL,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchSe, moduleTorchL], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchL, wArithm, moduleTorchSe,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchL, wArithm, moduleTorchSe,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchL, moduleTorchSe], 
                  moduleTorchCt, name='cat')

#Cn,Se; Se,Cn;
pset.addPrimitive(add, [moduleTorchCn, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchCn, wArithm, moduleTorchSe, wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchCn, moduleTorchSe], 
                  moduleTorchCt, name='cat')

pset.addPrimitive(add, [moduleTorchSe, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='add')
pset.addPrimitive(sub, [moduleTorchSe, wArithm, moduleTorchCn,  wArithm], 
                  moduleTorchCt, name='sub')
pset.addPrimitive(cat, [moduleTorchSe, moduleTorchCn], 
                  moduleTorchCt, name='cat')

# Recalibarion Layer Optional: Squeeze and Excitation Operation
pset.addPrimitive(se, [moduleTorchL],
                  moduleTorchSe, name='se')
pset.addPrimitive(se, [moduleTorchCn],
                  moduleTorchSe, name='se')

#Feature Connection Layer Optional
pset.addPrimitive(dense_connection, [moduleTorchL, tetha],#, kernelSizeConv, kernelSizeConv
                  moduleTorchCn, name='dCon')
pset.addPrimitive(res_connection, [moduleTorchL],
                  moduleTorchCn, name='rCon')
   
#Feature Extraction Layer
pset.addPrimitive(convolution, [moduleTorch, outChConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorch, name='conv')
pset.addPrimitive(sep_convolution, [moduleTorch, outChSConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorch, name='sconv')

pset.addPrimitive(convolution, [moduleTorch, outChConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorchL, name='conv')
pset.addPrimitive(sep_convolution, [moduleTorch, outChSConv, kernelSizeConv, kernelSizeConv, dilationRate],
                  moduleTorchL, name='sconv')

#Terminals
pset.addEphemeralConstant('outChConv', partial(lambda:outChConv()) , outChConv)
pset.addEphemeralConstant('outChSConv', partial(lambda:outChSConv()) , outChSConv)
pset.addEphemeralConstant('ksConv', partial(lambda:kernelSizeConv()) , kernelSizeConv)
pset.addEphemeralConstant('dilationR', partial(lambda:dilationRate()) , dilationRate)
pset.addEphemeralConstant('tetha', partial(lambda:tetha()), tetha)
pset.addEphemeralConstant('w', partial(lambda:wArithm()) , wArithm)

pset.renameArguments(ARG0="mod")

#%%Creator
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", stgp.PrimitiveTree, fitness=creator.FitnessMin)

#%%Toolbox
toolbox = base.Toolbox()
toolbox.register("expr", stgp.genHalfAndHalfMD, pset=pset, min_=2, max_=7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", stgp.compile, pset=pset)

toolbox.register('make_model', make_model, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("selectElitism", tools.selBest)

toolbox.register("mate", stgp.cxOnePointLeafBiased, termpb=0.1)

toolbox.register("expr_mut", stgp.genFull, min_=1, max_= 6)#6
toolbox.register("mutate_uniform", stgp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", stgp.mutEphemeral, mode='all')
toolbox.register("mutate_shrink", stgp.mutShrink)
toolbox.register("mutate_replace", stgp.mutNodeReplacement, pset=pset)

toolbox.decorate("mate", stgp.staticLimit(key=operator.attrgetter("height"), max_value=15))
toolbox.decorate("mutate_eph", stgp.staticLimit(key=operator.attrgetter("height"), max_value=15))
toolbox.decorate("mutate_shrink", stgp.staticLimit(key=operator.attrgetter("height"), max_value=15))
toolbox.decorate("mutate_replace", stgp.staticLimit(key=operator.attrgetter("height"), max_value=15))
toolbox.decorate("mutate_uniform", stgp.staticLimit(key=operator.attrgetter("height"), max_value=15))

#toolbox.register("save_ind", save_ind)
toolbox.register("plt_ind", plt_ind)
    
# toolbox.register("save_graphtvd", save_graphtvd)    
# toolbox.register("save_graphtv", save_graphtv)
toolbox.register("identifier", identifier, length=20)


#%%folder storage
foldername = 'humai-test'
"""Create folder to storage"""
path= "C:/Users/josef/OneDrive/Documentos/DIA/Talleres/Taller-HumaiAR2026/"
ruta=path+str(foldername)
if not os.path.exists(ruta):
    os.makedirs(ruta)

#%%Data
path_images = "C:/Users/josef/OneDrive/Documentos/DIA/NASGPNet/comparison_datasets/images_DRIVE"
in_channels = 3
out_channels = 2
dataset_type = Dataset2D
no_classes_msk = 2
image_height = 256
image_width = 256
batch_size = 1

"""Split Data (Percent or Static, 70-15-15)"""
train_set, valid_set, test_set = dataSplit.get_data(0.7, 0.15, 0.15, path_images,_format='.png')

#%%Training parameters
nepochs = 1#
lossfn = DiceLoss(average='macro', include_background=False, softmax=False, eps=1e-6)
metrics = [
    DiceMetric(average='macro', include_background=False, softmax=False, eps=1e-6),
    ]
# mstats=statics_(metrics)
lr = 0.001#0.0001
tolerance = 0 # for early stopping
verbose_train=False
device='cuda:0'
save_model=False
save_images=False
save_data=False

max_params = 31038000
w = 0.01

#%%Evaluation function
from objective_functions import evaluate_NoParameters, evaluate_Segmentation
from toolbox_functions import _get_metric_keys, _save_evaluation_data
from data.dataloader import loaders

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

toolbox.register("evaluate", evaluation,
                            nepochs=nepochs,
                            tolerance=tolerance,
                            lossfn=lossfn,
                            metrics=metrics,
                            lr=lr,
                            dataset_type = dataset_type,
                            no_classes_msk = no_classes_msk,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            batch_size = batch_size,
                            image_height = image_height,
                            image_width = image_width,
                            max_params=max_params, w=w,
                            train_set=train_set,
                            valid_set=valid_set,
                            test_set=test_set,
                            pset = pset,
                            device=device,
                            ruta=ruta,
                            verbose_train=verbose_train,
                            save_model=save_model,
                            save_images=save_images,
                            save_data=save_data
                            )

#%%Statistics
import numpy as np
"""Statistics of DEAP"""
def statics_(metrics):
    # Statics
    dict_stats = {"Fitness":tools.Statistics(lambda individual: individual.fitness.values),
                  "Size":tools.Statistics(len),
                  "Depth":tools.Statistics(lambda individual: individual.height),
                  "Params":tools.Statistics(lambda individual: individual.params)}
    
    for m in metrics:
        key = str(type(m)).strip('>').strip("'").split('.')[-1]
        dict_stats[key]=tools.Statistics(lambda individual: getattr(individual, key))
            
    mstats = tools.MultiStatistics(dict_stats)
    
    mstats.register("mean", np.mean)
    mstats.register("median", np.median)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    return mstats

mstats = statics_(metrics)

#%%Variation operators
import random

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


#%%NASGP-Net Algorithm
from datetime import datetime, timedelta

def nasgpnet(pop_size, toolbox, 
             # pset,
             cxpb, mutpb, ngen, nelit,
             ruta,  
             stats = None, halloffame=None, verbose_evo=__debug__):
    
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    metrics = toolbox.evaluate.keywords["metrics"]

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
    # idx=0
    no_evs=0
    cache={}
    
    # ###Individuals to evaluate
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    total_inds = len(invalid_ind)
    
    for idx in range(total_inds):#len(invalid_ind)):
        ind = invalid_ind[idx]
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
            
        #time
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t
        
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
    
    for gen in range(1, ngen+1):
        print('\nGen:\t', gen)
        
        #elitism
        elitism_inds = toolbox.selectElitism(toolbox.clone(population), k=nelit)

        #tournament
        offspring = toolbox.select(population, len(population) - nelit)
        
        #crossover and mutation
        offspring = MateMutation(offspring, toolbox, len(offspring), cxpb, mutpb) #4

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        total_inds = len(invalid_ind)
        
        #while len(invalid_ind)>0:
        for idx in range(total_inds): #range(len(invalid_ind)):
            ind = invalid_ind[idx]
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
            init_time = t
            
            #update the population every evaluation in order to keep it in the checkpoint
            population = elitism_inds+offspring
            
            print(f"{idx}/{total_inds}", gen, ruta.split("/").pop(), delta_t)
            
        if halloffame is not None:
            halloffame.update(offspring)
            
        #time
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t #Keep delta_t, no matters loose t
        
        #best ind
        best_ind = tools.selBest(population, 1)[0]
        print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3), best_ind.params, gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        dict_log = {"gen":gen,
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


#%%Calling nasgpnet
pop_size = 15
ngen = 5
cxpb = 0.5
mutpb = 0.49
nelit = 1
hof = tools.HallOfFame(nelit)
verbose_evo = False

pop, log, cache = nasgpnet(pop_size, 
                           toolbox, 
                           cxpb, 
                           mutpb, 
                           ngen, 
                           nelit, 
                           ruta, 
                           stats = mstats,
                           halloffame = hof,
                           verbose_evo = verbose_evo)

#%%Graficas de convergencia, guardado de estadísticas y análisis de funciones
from utils.deap_utils import functionAnalysis, log2csv
import matplotlib.pyplot as plt

##Convergence graph
def convergence_graph(logbook, ruta):
    gen=logbook.select("gen")
    fit_max=logbook.chapters["Fitness"].select("max")
    fig, host = plt.subplots()
    p1, = host.plot(gen, fit_max, "b-", label="Max Fit")
    host.set_xlabel("Generations")
    host.set_ylabel("Fitness")
    host.yaxis.label.set_color(p1.get_color())
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    lines = [p1]

    host.legend(lines, [l.get_label() for l in lines], loc="center right")

    plt.close(fig)
    plt.show()
    fig.savefig(ruta+"/Convergencia.png")

#Show and save Statics as .csv
log2csv(log, mstats, ruta)

#Plot Convergence Graphs
convergence_graph(log, ruta)

#Function Analysis
functionAnalysis(pop, 5, pset, ruta)

#%%Save evolved population and cache
from utils.saving_utils import save_population_csv, save_cache_csv
metrics_names = _get_metric_keys(metrics)
save_population_csv(
        pop,
        os.path.join(ruta, "population.csv"),
        metrics_names
    )

save_cache_csv(
        cache,
        os.path.join(ruta, "cache.csv"),
        metrics_names
    )


#%%Select the best individual
best = tools.selBest(pop, 1)[0]

#Retrain the best individual
nepochs=100
verbose_train=True
save_model=True
save_images=True
save_data=True

fit, params, dice = toolbox.evaluate(best,
                      nepochs=nepochs,
                      tolerance=tolerance,
                      verbose_train=verbose_train,
                      save_model=save_model,
                      save_images=save_images,
                      save_data=save_data)

#assign attributes to the individual
best.fitness.values = fit,
best.params = params
best.DiceMetric = dice


#%%Save best individual as csv
from utils.saving_utils import save_best_csv

save_best_csv(
    best,
    os.path.join(ruta, "best.csv"),
    metrics_names
    )

# #%% Save execution
# from utils.save_utils import save_execution
# save_execution(ruta, foldername, pop, log, cache, str(best))

#%%Recover data#!!!
from utils.saving_utils import load_population_csv, load_cache_csv, load_best_csv
pop_loaded = load_population_csv(
    filepath=os.path.join(ruta, "population.csv"),
    creator=creator,
    pset=pset,
    stgp_module=stgp,
    metrics_names=metrics_names
)

cache_loaded = load_cache_csv(
    filepath=os.path.join(ruta, "cache.csv"),
    creator=creator,
    pset=pset,
    stgp_module=stgp,
    metrics_names=metrics_names
)

best_loaded = load_best_csv(
    filepath=os.path.join(ruta, "best.csv"),
    creator=creator,
    pset=pset,
    stgp_module=stgp,
    metrics_names=metrics_names
)