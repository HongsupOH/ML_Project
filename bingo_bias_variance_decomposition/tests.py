from bingo.symbolic_regression.explicit_regression import ExplicitRegression
#from bingo.symbolic_regression.implicit_regression import ImplicitRegression
from implicit_regression_bvd import ImplicitRegression

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression import ExplicitTrainingData
from bingo.symbolic_regression import ImplicitTrainingData

from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evaluation.evaluation import Evaluation

from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file
from bingo.evolutionary_optimizers.evolutionary_optimizer import load_evolutionary_optimizer_from_file

from bias_variance_decomp import bias_variance_decomposition

from helper import gen_data

import time
import numpy as np
import math as m
from mpi4py import MPI
import sys
import json
import logging

import argparse

from helper_funcs import *

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

def create_evolutionary_optimizer(operators, hyperparams, checkpoint_file):
    pop_size = hyperparams["pop_size"]
    stack_size = hyperparams["stack_size"]
    differential_weight = hyperparams["differential_weight"]
    crossover_rate = hyperparams["crossover_rate"]
    mutation_rate = hyperparams["mutation_rate"]
    evolution_algorithm = hyperparams["evolution_algorithm"]

    rank = MPI.COMM_WORLD.Get_rank()
    n = 10
    data = gen_data(n)
    X,Y = data[:,:-1],data[:,-1]
    
    training_data = ExplicitTrainingData(X, Y)
    #training_data = ImplicitTrainingData(data)

    component_generator = ComponentGenerator(input_x_dimension = training_data.x.shape[1])
    for opp in operators:
        component_generator.add_operator(opp)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    
    agraph_generator = AGraphGenerator(stack_size, component_generator,use_simplification = False)
    
    #fitness = ExplicitRegression(training_data=training_data)
    #fitness = ImplicitRegression(training_data=training_data)
    fitness = bias_variance_decomposition(training_data=training_data)

    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='L-BFGS-B')
    local_opt_fitness.optimization_options = {'options':{'gtol':1e-12, 'ftol':1e-12, 'maxls':50,
                                                        'maxiter':10000, 'maxfun':10000},
                                              'tol':1e-12}

    local_opt_fitness.param_init_bounds = [-0.1, 0.1]
    
    evaluator = Evaluation(local_opt_fitness)
    
    if evolution_algorithm == "DeterministicCrowding":
        ea = DeterministicCrowdingEA(
            evaluator, crossover, mutation, crossover_rate, mutation_rate)
    elif evolution_algorithm == "AgeFitness":
        ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                          crossover_rate, mutation_rate, pop_size)

    island = Island(ea, agraph_generator, pop_size)

    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)

    return ParallelArchipelago(island, hall_of_fame=pareto_front)

def main(experiment_params, checkpoint=None):
    rank = MPI.COMM_WORLD.Get_rank()

    log_file = experiment_params["log_file"]
    checkpoint_file = experiment_params["checkpoint_file"]

    logging.basicConfig(filename=f"{log_file}_{rank}.log",
                        filemode="a", level=logging.INFO)

    hyperparams = experiment_params["hyperparams"]
    operators = experiment_params["operators"]

    if checkpoint is None:

        optimizer = create_evolutionary_optimizer(operators, hyperparams, checkpoint_file)

    else:
        optimizer = checkpoint

    max_generations = hyperparams["max_generations"]
    min_generations = hyperparams["min_generations"]
    fitness_threshold = hyperparams["fitness_threshold"]
    stagnation_threshold = hyperparams["stagnation_threshold"]
    check_frequency = hyperparams["check_frequency"]

    print("Starting evolution...")
    # go do the evolution and send back the best equations
    optim_result = optimizer.evolve_until_convergence(max_generations, fitness_threshold,
                                                      convergence_check_frequency=check_frequency, min_generations=min_generations,
                                                      stagnation_generations=stagnation_threshold, checkpoint_base_name=checkpoint_file, num_checkpoints=1)

    if rank == 0:
        pareto_front = optimizer.hall_of_fame
        print(optim_result)
        print("Generation: ", optimizer.generational_age)
        print_pareto_front(pareto_front)
        log_trial(result_file, operators, hyperparams, pareto_front, optim_result)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", dest="experiment_file",
                        type=str, help="Experiment file to run", required=True)
    parser.add_argument("-n", required=False, dest="experiment_idx",
                        help="Index of experiment in give experiment file", type=int)
    parser.add_argument(
        "-k", help="Checkpoint file to resume from", type=str, dest="checkpoint_file")

    args = parser.parse_args()

    if args.checkpoint_file is not None:
        checkpoint = load_parallel_archipelago_from_file(args.checkpoint_file)
    else:
        checkpoint = None

    with open(args.experiment_file) as f:
        setup_json = json.load(f)

    if type(setup_json) is list:
        if args.experiment_idx is None:
            for setup in setup_json:
                main(setup)
        else:
            main(setup_json[args.experiment_idx], checkpoint)
    else:
        main(setup_json, checkpoint)
