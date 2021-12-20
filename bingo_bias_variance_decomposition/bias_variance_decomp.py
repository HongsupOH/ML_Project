import numpy as np

import logging

from bingo.evaluation.fitness_function import VectorBasedFunction

LOGGER = logging.getLogger(__name__)

class bias_variance_decomposition(VectorBasedFunction):

    def __init__(self,
                 training_data,
                 metric= "mae",
                 wb = 0.3,
                 wv = 0.7):

        super().__init__(training_data, metric)
        
        self.wb = wb
        self.wv = wv
        

    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        m = self.training_data.x.shape[0]
        
        y_hat = individual.evaluate_equation_at(self.training_data.x).reshape(m)
        y = self.training_data.y.reshape(m)
        #Cd = sum(abs(y_hat - y)/(abs(y_hat) + abs(y)))
        Cd = np.mean(np.abs(y_hat - y)/(np.abs(y_hat) + np.abs(y)))         

        Cs_s = []
        for _ in range(10):
            ind = np.random.choice(m,m,replace=True)
            xs = self.training_data.x[ind]
            ys = self.training_data.y[ind].reshape(m)

            ys_hat = individual.evaluate_equation_at(xs).reshape(m)
            Cs = np.mean(np.abs(ys_hat - ys)/(np.abs(ys_hat) + np.abs(ys)))
            #Cs = sum(abs(ys_hat - ys)/(abs(ys_hat) + abs(ys)))
            Cs_s.append(Cs)

        Cs_s = np.array(Cs_s)
        Cs_avg = np.mean(Cs_s)

        #VarD = sum((Cs_s - Cs_avg)**2)/len(Cs_s)
        VarD = np.mean((Cs_s - Cs_avg)**2)
        fitness = self.wb*Cd + self.wv*VarD
            
        return fitness
