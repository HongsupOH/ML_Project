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
       size,trial = info[2],info[-2]
       b,v = info[3],info[4]
       cur_gen = file.split('_')[-1].split('.')[0]
      
       if (b,v) not in data:
          data[(b,v)] = {}
          data[(b,v)]['fitness'] = []
          data[(b,v)]['complexity'] = []
          data[(b,v)]['generation'] = []
          data[(b,v)]['size'] = size
       archipelago = load_parallel_archipelago_from_file(path+'/'+file)
       hof = archipelago.hall_of_fame
       if len(hof)!=0:
            data[(b,v)]['fitness'].append(hof[0].fitness)
            data[(b,v)]['complexity'].append(hof[0].get_complexity())
            data[(b,v)]['generation'].append(int(cur_gen)) 
   return data
 

def distribution_plot(data):
    weights = list(data)
    colors = ['blue','red','green','orange']
    for weight in weights:
        plt.figure()
        dist = data[weight]['complexity']
        fit = np.array(data[weight]['fitness'])
        gen = np.array(data[weight]['generation'])
        unique = list(set(dist))
        size = data[weight]['size']
        print("#### bias-{},variacne-{} ####".format(weight[0],weight[1]))
        for ele in unique:
            print("Complexity {} - count is {}".format(ele,dist.count(ele)))
        print("Average fitness {}".format(np.median(fit)))
        print("Average generation {}".format(np.median(gen)))

path = 'checkpoints_st_40'
files = [f for f in listdir(path) if isfile(join(path, f))]
data = plot_gen_fit(path,files)
distribution_plot(data)
