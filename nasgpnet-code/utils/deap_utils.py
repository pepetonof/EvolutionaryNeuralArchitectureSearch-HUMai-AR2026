# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 13:25:01 2021

@author: josef

Utils for Genetic Programming
"""

from deap import tools
from deap.gp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import isnan, isinf

"""Functions to create multistatics and manipulate 'log' variable after evolutionary process"""

#%%Statics for evolutionary process
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
    
def plot_surrogate(log, ruta):
    
    gen = log.select("gen")
    nevls = log.select("nevals")
    train_samples = log.select("nevals")
    test_samples = [train_samples[i]-train_samples[i-1] for i in range(1, len(train_samples))]
    test_samples.insert(0,0)
    test_samples = np.array(test_samples)
    train_samples.insert(0, train_samples[0])
    train_samples = np.array(train_samples[:-1])
    # total_samples = np.array(log.select("nevals")) #samples to train the surrogate model
    # train_proportion = train_samples/total_samples
    # test_proportion = test_samples/total_samples
    
    in_cache = log.select("nevals")
    new_ones = log.select("new_inds")

    new_ones[0]=0
    
    in_cache = np.array(in_cache)
    new_ones = np.array(new_ones)

    # in_cache_ratio = (in_cache - new_ones)/in_cache
    new_ones_ratio = new_ones#/in_cache
    
    
    colors=list(mcolors.TABLEAU_COLORS.keys())[:8]
    _pearson = log.select("pearson")
    _spearman = log.select("spearman")
    _tau = log.select("kendall_tau")
    _r2 = log.select("coef_r2")
    # _pk = log.select("precisionK")
    # _mse = log.select("mse")
    _time = log.select("t_train")
    
    fig, ax = plt.subplots(2, 4, sharex=False, figsize=(22,8)) 
    flag = False
    _reconstructed = [x is None for x in _spearman]
    for i, (testp, f) in enumerate(zip(new_ones_ratio, _reconstructed)): #in_cache_ratio,
        alpha_val = 0.2 if f or testp==0 else 0.4  # Hacer la barra transparente si la proporción óptima es 1
        ax[0,0].bar(gen[i], testp, color='orange', alpha=alpha_val, label='Not in cache' if testp>0 and not flag else "")   
        # ax[0,0].bar(gen[i], trainp, bottom=testp, color='green', alpha = alpha_val, label='In cache' if testp>0 and not flag else "")
        flag = True if testp>0 and not flag else flag
     
    #nevals
    # ax[0,0].set_xlabel('Generations')
    ax[0,0].set_ylabel('No. of individuals', fontsize=12)
    # ax[0,0].set_ylim(0, 1)
    ax[0,0].legend(loc='upper left')
    # ax[0,0].set_yticks(np.arange(0, 1.01, 0.1))
    ax[0,0].grid(axis = 'y', linestyle='--')
    ax[0,0].tick_params(axis='x', labelsize=14)
    ax[0,0].tick_params(axis='y', labelsize=14)
    # ax[0,0].set_xticklabels([])
    
    # Gráfico de línea superpuesto
    ax2 = ax[0,0].twinx()
    ax2.plot(gen, nevls, 'b-', label='No. Evaluations', marker = 'o')
    ax2.set_ylabel('', color='blue')
    ax2.tick_params('y', colors='blue')
    ax2.set_ylabel('No. Evaluations', fontsize=12)

    ax[0,0].yaxis.set_label_position('right')
    ax[0,0].yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    # ax2.set_xlim([gen[0]-0.4, gen[-1]+0.4])
    # ax2.set_xticks(np.arange(0,gen[-1]+1,2))
    # ax2.tick_params(axis='x', labelsize=14)
    # ax2.tick_params(axis='y', labelsize=14)
    
    #convergence graph
    mean_fit = np.array(log.chapters["Fitness"].select("mean"))
    max_fit = np.array(log.chapters["Fitness"].select("max"))
    std_fit = np.array(log.chapters["Fitness"].select("std"))
    ax[0,1].plot(gen, mean_fit, color="blue", label=r"$fit_{mean}$", linewidth=2, marker="o")
    ax[0,1].fill_between(gen, mean_fit-std_fit, mean_fit+std_fit, color="blue", alpha=0.2, label = "std")
    ax[0,1].tick_params(axis='x', labelsize=14)
    ax[0,1].tick_params(axis='y', labelsize=14)
    ax[0,1].set_ylabel("Fitness", fontsize=12)
    ax[0,1].grid(axis = 'y', linestyle='--')
    # ax[0,1].legend(loc="lower right")
    xticks_from_plot = ax[0,1].get_xticks()[1:-1]
    ax[0,0].set_xticks(xticks_from_plot)
    
    
    ax[0,1].plot(gen, max_fit, color="red", label=r"$fit_{best}$", linewidth=2, marker="o")
    ax[0,1].legend(loc='lower right')
    # ax2 = ax[0, 1].twinx()
    # ax2.plot(gen, max_fit, color="red", linewidth=2, marker="o")
    
    # ax2.set_ylabel('', color = 'red')
    # ax2.tick_params('y', colors = 'red')
    # ax2.set_ylabel('Max Fitness', fontsize=14)
    # ax2.yaxis.set_label_position('right')
    # ax2.yaxis.set_ticks_position('right')
    # ax2.tick_params(axis='x', labelsize=14)
    # ax2.tick_params(axis='y', labelsize=14)
    
    
    #train_time
    markerline, stemlines, baselines = ax[0,2].stem(gen, _time, linefmt=colors[1], markerfmt='D')
    markerline.set_markerfacecolor(colors[1])
    markerline.set_markeredgecolor(colors[1])
    ax[0,2].tick_params(axis='x', labelsize=14)
    ax[0,2].tick_params(axis='y', labelsize=14)
    ax[0,2].set_ylabel(r"$\mathbf{t_{train} (s)}$")
    ax[0,2].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    ax[0,2].grid(axis = 'y', linestyle='--')
    # ax[0,1].set_xticklabels([])
    
    #precisionK
    # markerline, _, _ = ax[0,3].stem(gen, _pk, linefmt=colors[2], markerfmt='D')
    # markerline.set_markerfacecolor(colors[2])
    # markerline.set_markeredgecolor(colors[2])
    # ax[0,3].tick_params(axis='x', labelsize=14)
    # ax[0,3].tick_params(axis='y', labelsize=14)
    # ax[0,3].set_ylabel(r"$\mathbf{P@K}$")
    # ax[0,3].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    # ax[0,3].grid(axis = 'y', linestyle='--')
    
    ax[0, 3].axis('off')
    
    # ax[0,2].set_xticklabels([])
    
    #mse
    # markerline, _, _ = ax[0,3].stem(gen, _mse, linefmt=colors[3], markerfmt='D')
    # markerline.set_markerfacecolor(colors[3])
    # markerline.set_markeredgecolor(colors[3])
    # ax[0,3].tick_params(axis='x', labelsize=14)
    # ax[0,3].tick_params(axis='y', labelsize=14)
    # ax[0,3].set_ylabel(r"$\mathbf{mse}$")
    # ax[0,3].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    # ax[0,3].grid(axis = 'y', linestyle='--')
    # # ax[0,3].set_xticklabels([])
    
    #pearson
    markerline, _, _ = ax[1,0].stem(gen, _pearson, linefmt=colors[4], markerfmt='D',)
    markerline.set_markerfacecolor(colors[4])
    markerline.set_markeredgecolor(colors[4])
    ax[1,0].tick_params(axis='x', labelsize=14)
    ax[1,0].tick_params(axis='y', labelsize=14)
    ax[1,0].set_ylabel(r"$\mathbf{r}$")
    ax[1,0].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    ax[1,0].set_ylim(-1.1,1.1)
    ax[1,0].set_yticks(np.arange(-1.0,1.02,0.2))
    ax[1,0].grid(axis = 'y', linestyle='--')
    ax[1,0].set_xlabel("Generations", fontweight="bold")
    
    #non computable pearson  
    nan_idx = [i for i,v in enumerate(_pearson) if isinstance(v, float) and isnan(v)]
    _nan = [gen[i] for i in nan_idx]
    if len(_nan)>0:
        ax[1,0].plot(
                _nan,                
                [0]*len(_nan),      #
                marker='x',            
                markersize=8,          
                linestyle='',           
                color='red',           
                transform=ax[1,0].get_xaxis_transform(), 
                zorder=5,              
                label='' 
            )
    
    #r2
    # ax[1,1].plot(gen, _r2, color=colors[3], linewidth=2.0)
    markerline, _, _ = ax[1,1].stem(gen, _r2, linefmt=colors[5], markerfmt='D')
    markerline.set_markerfacecolor(colors[5])
    markerline.set_markeredgecolor(colors[5])
    ax[1,1].tick_params(axis='x', labelsize=14)
    ax[1,1].tick_params(axis='y', labelsize=14)
    ax[1,1].set_ylim(-1.1,1.1)
    ax[1,1].set_yticks(np.arange(-1.0,1.02,0.2))
    ax[1,1].set_ylabel(r"$\mathbf{R^2}$")
    ax[1,1].set_xlabel("Generations", fontweight="bold")
    ax[1,1].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    # ax[1,1].set_yticks(np.arange(-1.0,1.02,0.2))
    ax[1,1].grid(axis = 'y', linestyle='--')
    #non computable tau
    nan_idx = [i for i,v in enumerate(_r2) if isinstance(v, float) and v<0 and isinf(v)]
    _nan = [gen[i] for i in nan_idx]
    
    if len(_nan)>0:
        ax[1,1].plot(
                _nan,                
                [0]*len(_nan),      #
                marker='x',            
                markersize=8,          
                linestyle='',           
                color='red',           
                transform=ax[1,1].get_xaxis_transform(), 
                zorder=5,              
                label='' 
            )
    
    
    #spearman
    markerline, _, _ = ax[1,2].stem(gen, _spearman, linefmt=colors[6], markerfmt='D')
    markerline.set_markerfacecolor(colors[6])
    markerline.set_markeredgecolor(colors[6])
    ax[1,2].tick_params(axis='x', labelsize=14)
    ax[1,2].tick_params(axis='y', labelsize=14)
    ax[1,2].set_ylabel(r"$\mathbf{\rho}$")
    ax[1,2].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    ax[1,2].set_ylim(-1.1,1.1)
    ax[1,2].set_yticks(np.arange(-1.0,1.02,0.2))
    ax[1,2].grid(axis = 'y', linestyle='--')
    ax[1,2].set_xlabel("Generations", fontweight="bold")
    
    #non computable spearman
    nan_idx = [i for i,v in enumerate(_spearman) if isinstance(v, float) and isnan(v)]
    _nan = [gen[i] for i in nan_idx]
    
    if len(_nan)>0:
        ax[1,2].plot(
                _nan,                
                [0]*len(_nan),      #
                marker='x',            
                markersize=8,          
                linestyle='',           
                color='red',           
                transform=ax[1,2].get_xaxis_transform(), 
                zorder=5,              
                label='' 
            )
    
    
    #tau
    # ax[0,3].plot(gen, _tau, color=colors[2], linewidth=2.0)
    markerline, _, _ = ax[1,3].stem(gen, _tau, linefmt=colors[7], markerfmt='D')
    markerline.set_markerfacecolor(colors[7])
    markerline.set_markeredgecolor(colors[7])
    ax[1,3].tick_params(axis='x', labelsize=14)
    ax[1,3].tick_params(axis='y', labelsize=14)
    ax[1,3].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    ax[1,3].set_ylim(-1.1,1.1)
    ax[1,3].set_ylabel(r"$\mathbf{\tau}$")
    ax[1,3].set_xlim([gen[0]-0.5, gen[-1]+0.5])
    ax[1,3].set_yticks(np.arange(-1.0,1.02,0.2))
    ax[1,3].grid(axis = 'y', linestyle='--')
    ax[1,3].set_xlabel("Generations", fontweight="bold")
    
    #non computable tau
    nan_idx = [i for i,v in enumerate(_tau) if isinstance(v, float) and isnan(v)]
    _nan = [gen[i] for i in nan_idx]
    
    if len(_nan)>0:
        ax[1,3].plot(
                _nan,                
                [0]*len(_nan),      #
                marker='x',            
                markersize=8,          
                linestyle='',           
                color='red',           
                transform=ax[1,3].get_xaxis_transform(), 
                zorder=5,              
                label='' 
            )
    
    fig.tight_layout()
    plt.close(fig)
    plt.show()
    fig.savefig(ruta+"/Train-Test-Samples2.png", dpi=300)

    
def plot_nevals(log, ruta):
    gen = log.select("gen")
    nevals = log.select("nevals")
    fig, host = plt.subplots()
    p1, = host.plot(gen, nevals, "b-", label="No. Evals")
    
    host.set_xlabel("Generations")
    host.set_ylabel("No. Evals")
    host.yaxis.label.set_color(p1.get_color())
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    lines = [p1]

    host.legend(lines, [l.get_label() for l in lines], loc="center right")
    
    plt.close(fig)
    plt.show()
    fig.savefig(ruta+"/NoEvals.png", dpi =300)

#%%Graph convergence, size and depth of evolutionary process as .png 
def show_statics(estadisticas, rutita):
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    
    def convergence_graph2():
        gen=estadisticas.select("gen")
        fit_min=estadisticas.chapters["Fitness"].select("max")
        fig, host = plt.subplots()
        p1, = host.plot(gen, fit_min, "b-", label="Max Fit")
        host.set_xlabel("Generations")
        host.set_ylabel("Fitness")
        host.yaxis.label.set_color(p1.get_color())
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        lines = [p1]
        
        host.legend(lines, [l.get_label() for l in lines], loc="center right")
        
        plt.close(fig)
        plt.show()
        fig.savefig(rutita+"/Convergencia2.png")
        
    
    def convergence_graph():
        gen=estadisticas.select("gen")
        fit_min=estadisticas.chapters["Fitness"].select("max")
        size_avgs=estadisticas.chapters["Size"].select("avg")
        depth_avgs=estadisticas.chapters["Depth"].select("avg")
        
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        par1 = host.twinx()
        par2 = host.twinx()
        
        par2.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)
        
        p1, = host.plot(gen, fit_min, "b-", label="Max Fit")
        p2, = par1.plot(gen, size_avgs, "r-", label="Avg Size")
        p3, = par2.plot(gen, depth_avgs, "g-", label="Avg Depth ")
        
        host.set_xlabel("Generations")
        host.set_ylabel("Fitness")
        par1.set_ylabel("Size Avg")
        par2.set_ylabel("Depth Avg")
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [p1, p2, p3]
        
        host.legend(lines, [l.get_label() for l in lines], loc="center right")
        
        plt.close(fig)
        plt.show()
        fig.savefig(rutita+"/Convergencia.png")
    
    def metrics():
        gen=estadisticas.select("gen")
        dice=estadisticas.chapters["DiceMetric"].select('mean')
        iou=estadisticas.chapters["IoUMetric"].select('mean')
        hds=estadisticas.chapters["HD95Metric"].select('mean')
        nsd=estadisticas.chapters["NSDMetric"].select('mean')
        
        fig, host = plt.subplots()
        par1=host.twinx()
        
        p1, = host.plot(gen, dice, "b-", label="Dice")
        p2, = host.plot(gen, iou, "r-", label="IoU")
        p3, = par1.plot(gen, hds, "g-", label="HD95")
        p4, = par1.plot(gen, nsd, "m-", label="NSD")
        host.set_xlabel("Generations")
        host.set_ylabel("Overlap", color="k")
        par1.set_ylabel("Distance", color="g")
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p3.get_color())
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p3.get_color(), **tkw)
        
        lines = [p1, p2, p3, p4]
        host.legend(lines, [l.get_label() for l in lines], loc="lower right")
        
        plt.close(fig)
        plt.show()
        fig.savefig(rutita+"/Metricas.png")
        
    def size_depth():
        gen=estadisticas.select("gen")
        size_avgs=estadisticas.chapters["Size"].select("mean")
        depth_avgs=estadisticas.chapters["Depth"].select("mean")
        
        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)
        
        par1 = host.twinx()
        # par2 = host.twinx()
        
        # par2.spines["right"].set_position(("axes", 1.2))
        # make_patch_spines_invisible(par2)
        # par2.spines["right"].set_visible(True)
        
        p1, = host.plot(gen, size_avgs, "b-", label="Avg Size")
        p2, = par1.plot(gen, depth_avgs, "r-", label="Avg Depth")
        
        host.set_xlabel("Generations")
        host.set_ylabel("Size")
        par1.set_ylabel("Depth")
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        
        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)
        
        lines = [p1, p2]
        host.legend(lines, [l.get_label() for l in lines], loc="center right")
        
        plt.close(fig)
        plt.show()
        fig.savefig(rutita+"/Size_Depth.png")
        
    convergence_graph()
    convergence_graph2()
    size_depth()
    # metrics()
    return

#%% Save as csv
def log2csv(log, mstats, ruta):    
    lst_keys=[]
    lst_vals=[]
    for h in log.header:
        if h in list(log.chapters.keys()):
            for sts in mstats[h].fields:
                # print(h, sts)
                lst_keys.append(h+"_"+sts)
                lst_vals.append(log.chapters[h].select(sts))
        else:
            lst_keys.append(h)
            lst_vals.append(log.select(h))
         
    df=pd.DataFrame.from_dict(dict(zip(lst_keys, lst_vals)))
    df.to_csv(ruta+'/logbook.csv', index=False)
            
    return

#%%Save statics of evolutionary process as csv
def save_statics(log, ruta):
    gen = log.select("gen")
    evaluations = log.select("nevals")
    time = log.select("time")
    
    best = log.select('best')
    best_dice = log.select('best_dice')
    best_iou = log.select('best_iou')
    best_hd = log.select('best_hd')
    best_hd95 = log.select('best_hd95')
    best_nds = log.select('best_nds')
    best_params = log.select('best_params')
    
    # r_2 = log.select('r_2')
    # mse = log.select('mse')
    # rmse = log.select('rmse')
    # mae = log.select('mae')
 
    fit_maxs = log.chapters["Fitness"].select("max")
    fit_mins=log.chapters["Fitness"].select("min")
    fit_prom=log.chapters["Fitness"].select("avg")
    fit_std=log.chapters["Fitness"].select("std")
    
    dice_maxs = log.chapters["Dice"].select("max")
    dice_min = log.chapters["Dice"].select("min")
    dice_avgs = log.chapters["Dice"].select("avg")
    dice_std = log.chapters["Dice"].select("std")
    
    params_maxs = log.chapters["Params"].select("max")
    params_min = log.chapters["Params"].select("min")
    params_avgs = log.chapters["Params"].select("avg")
    params_std = log.chapters["Params"].select("std")
    
    size_maxs = log.chapters["Size"].select("max")
    size_min = log.chapters["Size"].select("min")
    size_avgs = log.chapters["Size"].select("avg")
    size_std = log.chapters["Size"].select("std")
    
    depth_maxs = log.chapters["Depth"].select("max")
    depth_min = log.chapters["Depth"].select("min")
    depth_avgs = log.chapters["Depth"].select("avg")
    depth_std = log.chapters["Depth"].select("std")
    
    dict={'Generations':gen,
          'Evaluations':evaluations,
          'Time':time,
          'Best_Ind':best,
          'Best_Dice':best_dice,
          'Best_Params':best_params,
          # "R_2":r_2,
          # "mse":mse,
          # "rmse":rmse,
          # "mae":mae,
          
          'Fitness_max':fit_maxs,
          'Fitness_min':fit_mins,
          'Fitness_avg':fit_prom,
          'Fitness_std':fit_std,
          
          'Dice max':dice_maxs,
          'Dice min': dice_min,
          'Dice avg': dice_avgs,
          'Dice std': dice_std,
          
          'Param max': params_maxs,
          'Param min': params_min,
          'Params avg': params_avgs,
          'Params std' : params_std,
          
          'Size max':size_maxs,
          'Size min':size_min,
          'Size avg':size_avgs,
          'Size std':size_std,
          
          'Depth max':depth_maxs,
          'Depth min':depth_min,
          'Depth avg':depth_avgs,
          'Depth std':depth_std,
          }
    daf=pd.DataFrame.from_dict(dict)
    daf.to_csv(ruta+'/proceso_evolutivo.csv', index=False)
    return

#%%FunctionAnalysis 
def functionAnalysis(pop, n, pset, ruta):
    def dicts(lst):
        dic={}
        for b in lst:
            string=str(b)
            specialchar = '¿?¡!{}[]()<>\'""+-*/.:,;~…‘’“”``´´^¨#$%&_—°|¬1234567890«»×=//\\'
            ignore=set("mod".split())
            for c in specialchar:
                string=string.replace(c," ")
            string_lst=string.split()
            string_lst=list(word for word in string_lst if word not in ignore)
            for w in string_lst:
                if w in dic:
                    dic[w]+=1
                else:
                    dic[w]=1
        return dic
        
    bests=tools.selBest(pop,n)
    worsts=tools.selWorst(pop, n)
    dbests=dicts(bests)
    dworsts=dicts(worsts)
    # print(dbests,dworsts)
    funciones=list(pset.context.keys())[1:]
    # print(funciones)
    faltantes1=[f for f in funciones if f not in set(dbests.keys())]
    faltantes2=[f for f in funciones if f not in set(dworsts.keys())]
    if len(faltantes1)>0:
        for f in faltantes1:
            dbests[f]=0
    if len(faltantes2)>0:
        for f in faltantes2:
            dworsts[f]=0
        
    dbests=dict(sorted(dbests.items()))
    dworsts=dict(sorted(dworsts.items()))
    # print(faltantes1,faltantes2)
    # print(dbests)
    # print(dworsts)
    
    numero_de_grupos = len(dbests.values())
    indice_barras = np.arange(numero_de_grupos)
    ancho_barras =0.4
    plt.bar(indice_barras, dbests.values(), width=ancho_barras, label='Best ind')
    plt.bar(indice_barras + ancho_barras, dworsts.values(), width=ancho_barras, label='Worst ind')
    plt.legend(loc='best')
    plt.xticks(indice_barras + ancho_barras, dbests.keys())
    plt.ylabel('Frequency')
    plt.xlabel('Functions')
    plt.title('Best and worst {n} individuals'.format(n=n))
    plt.savefig(ruta+"/FunctionsBar.png")
    plt.close()
    plt.show()
    
    