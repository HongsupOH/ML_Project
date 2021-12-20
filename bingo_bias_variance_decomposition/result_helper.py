from os import listdir
from os.path import isfile, join
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file


def plot_gen_fit(path,files):
   data = {}
   for i,file in enumerate(files):
       info = file.split('_')
       size,trial = info[2],info[3]
       cur_gen = file.split('_')[-1].split('.')[0]
       if size not in data:
          data[size] = {}
       if size in data:
          if int(trial) not in data[size]:
              data[size][int(trial)] = []
       archipelago = load_parallel_archipelago_from_file(path+'/'+file)
       hof = archipelago.hall_of_fame
       if len(hof)!=0:
            data[size][int(trial)].append([int(cur_gen),hof[0].fitness,hof[0].get_complexity()])
       
   return data
 

def distribution_plot(data):
    sizes = list(data)
    sizes.sort()
    for size in sizes:
        cur_data = data[size]
        trials = list(cur_data)
        trials.sort()
        fit,com,gen = [],[],[]
        for trial in trials:
            sub_data =  np.array(cur_data[trial])
            #sub_data = np.sort(sub_data,axis = 0)
            row = sub_data[-1]
            fit.append(row[1])
            com.append(row[2])
            gen.append(row[0]) 
        unique = list(set(com))
        print("### size - {} ###".format(size))
        for ele in unique:
           print('Complexity {}: count - {}'.format(ele,com.count(ele)))
        fit = np.array(fit)
        gen = np.array(gen)
        print('Average fitness {}'.format(np.median(fit)))
        print('Average generation {}'.format(np.median(gen)))


def plot_fit(data):
    plt.figure(figsize=(8,10))
    sizes = list(data)
    sizes.sort()
    colors = ['blue','red','green','orange']
    for size in sizes:
        cur_data = data[size]
        trials = list(cur_data)
        trials.sort()
        fits = []
        coms = []
        gens = []
        for trial in trials: 
            cur_sub_data = np.array(cur_data[trial])
            c = colors[int(sizes.index(size))]
            fitnesses = cur_sub_data[:,1].tolist()
            complexity = cur_sub_data[:,2].tolist()
            gen = cur_sub_data[:,0].tolist()
            max_gen = max(gen)
            best_fit = min(fitnesses)
            best_com = complexity[fitnesses.index(best_fit)]
            plt.plot(cur_sub_data[:,1],color = c,label='Size-{}'.format(size))
         
            fits.append(best_fit)
            coms.append(best_com)
            gens.append(max_gen)
        fits = np.array(fits)
        coms = np.array(coms)
        gens = np.array(gens)
        medfit = np.median(fits)
        medcoms = np.median(coms)
        medgens = np.median(gens)
        print('Size{} - fitness:{}, complexity:{}, generation{}'.format(size,medfit,medcoms,medgens))
          
    plt.grid()
    plt.legend()
    plt.savefig('fitness_plot.jpg')

path = 'checkpoints_explicit'
files = [f for f in listdir(path) if isfile(join(path, f))]
data = plot_gen_fit(path,files)
distribution_plot(data)
#plot_fit(data)
