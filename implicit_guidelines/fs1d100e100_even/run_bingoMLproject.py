import argparse
import numpy as np
import h5py
from mpi4py import MPI
import os

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.parallel_archipelago \
    import ParallelArchipelago, load_parallel_archipelago_from_file
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.fitness_predictor_island \
    import FitnessPredictorIsland
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, ImplicitTrainingData

# logging
from bingo.util import log
log.configure_logging(verbosity=log.DETAILED_INFO,
                      module=False, timestamp=False)
import logging
LOGGER = logging.getLogger(__name__)

# MPI setup
COMM = MPI.COMM_WORLD
COMM_RANK = COMM.Get_rank()
COMM_SIZE = COMM.Get_size()

# Set up and run Bingo!--------------------------------------------------------
def run_gp(filename, step_size, num_repeats, checkpoint_name, max_generations, 
           abs_tolerance, stagnation_gens, fs_exponent,
           convergence_freq, equ_pop_size, equ_size, operators,
           pred_pop_size, pred_size_ratio, pred_comp_ratio,
           mutation_rate, crossover_rate, req_x):

    all_opt_results = []
    for i in range(num_repeats):
        cpi_name = checkpoint_name
        if step_size != 1:
            cpi_name += "_k" + str(step_size)
        if num_repeats > 1:
            cpi_name += "_" + str(i)
        stats_name = cpi_name + "_" + str(i) + ".stats"
        log_name = cpi_name + "_" + str(i) + ".log"

        opt_result = single_gp_run(filename, step_size, abs_tolerance,
                                   cpi_name, convergence_freq, crossover_rate,
                                   equ_pop_size, equ_size, max_generations,
                                   mutation_rate, operators, pred_comp_ratio,
                                   pred_pop_size, pred_size_ratio, fs_exponent,
                                   stagnation_gens, stats_name, log_name, 
                                   req_x)
        all_opt_results.append(opt_result)


def single_gp_run(filename, step_size, abs_tolerance,
                  checkpoint_name, convergence_freq,
                  crossover_rate, equ_pop_size, equ_size, max_generations,
                  mutation_rate, operators, pred_comp_ratio, pred_pop_size, 
                  pred_size_ratio, fs_exponent, stagnation_gens, stats_file, 
                  log_file, req_x):
    # update logging
    log.configure_logging(verbosity=log.DETAILED_INFO,
                          module=False, timestamp=False,
                          stats_file=stats_file, logfile=log_file)

    _log_bingo_params(abs_tolerance, convergence_freq, crossover_rate,
                      equ_pop_size, equ_size, max_generations, mutation_rate,
                      operators, pred_comp_ratio, pred_pop_size,
                      pred_size_ratio, stagnation_gens, checkpoint_name, req_x)
    
    data = load_data(filename, step_size, fs_exponent)
    training_data = ImplicitTrainingData(data)

    # generation
    component_generator = ComponentGenerator(data.shape[1])
    for op in operators:
        component_generator.add_operator(op)
    # agraph_generator = AGraphGenerator(equ_size, component_generator)
    agraph_generator = AGraphGenerator(equ_size, component_generator,
                                        use_simplification=True) # prunes unneccessary parts

    # variation
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    # mutation = AGraphMutation(component_generator, command_probability=0.2,
    #                   node_probability=0.2, parameter_probability=0.2,
    #                   prune_probability=0.2, fork_probability=0.2) 

    # evaluation
    fitness = ImplicitRegression(training_data=training_data,
                                 required_params=req_x)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='Nelder-Mead')
    evaluator = Evaluation(local_opt_fitness)

    # HOF
    def agraph_similarity(ag_1, ag_2):
        """a similarity metric between agraphs"""
        return ag_1.fitness == ag_2.fitness and \
            ag_1.get_complexity() == ag_2.get_complexity()

    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)

    # evolutionary optimizer
    ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                      crossover_rate, mutation_rate, equ_pop_size)
    island = FitnessPredictorIsland(ea, agraph_generator, equ_pop_size,
                                    predictor_population_size=pred_pop_size,
                                    predictor_size_ratio=pred_size_ratio,
                                    predictor_computation_ratio=pred_comp_ratio,
                                    )

    archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front,
                                      non_blocking=True)

    # run opt
    opt_result = archipelago.evolve_until_convergence(
            max_generations=max_generations,
            fitness_threshold=abs_tolerance,
            stagnation_generations=stagnation_gens,
            checkpoint_base_name=checkpoint_name,
            convergence_check_frequency=convergence_freq,
            num_checkpoints=5)

    if COMM_RANK == 0:
        logging.info(":::TRIAL RESULT:::")
        logging.info(str(opt_result))
        logging.info("\n")

    return opt_result

def _log_bingo_params(abs_tolerance, convergence_freq, crossover_rate,
                      equ_pop_size, equ_size, max_generations, mutation_rate,
                      operators, pred_comp_ratio, pred_pop_size,
                      pred_size_ratio, stagnation_gens, checkpoint_name,
                      req_x):
    if COMM_RANK == 0:
        logging.info(":::TRIAL %s:::", checkpoint_name)
        logging.info("BINGO PARAMS:")
        logging.info("  --GENERAL--")
        logging.info("  NUMBER OF CPUS = %d", COMM_SIZE)

        logging.info("  --CONVERGENCE--")
        logging.info("  MAX GENERATIONS = %d", max_generations)
        logging.info("  ABSOLUTE FITNESS TOLERANCE = %f", abs_tolerance)
        logging.info("  STAGNATION GENERATIONS = %d", stagnation_gens)
        logging.info("  CONVERGENCE CHECK FREQUENCY = %d", convergence_freq)

        logging.info("  --EQUATIONS--")
        logging.info("  TOTAL EQUATION POPULATION SIZE = %d",
                     COMM_SIZE * equ_pop_size)
        logging.info("  EQUATION POPULATION PER CPU = %d", equ_pop_size)
        logging.info("  EQUATION STACK SIZE = %d", equ_size)
        logging.info("  EQUATION OPERATORS = [%s]", ", ".join(operators))

        logging.info("  --FITNESS PREDICTORS--")
        logging.info("  PREDICTOR POPULATION PER CPU = %d", pred_pop_size)
        logging.info("  PREDICTOR SIZE RATIO = %f", pred_size_ratio)
        logging.info("  PREDICTOR COMP RATIO = %f", pred_comp_ratio)

        logging.info("  --EVOLUTIONARY ALGORITHM--")
        logging.info("  ALGORITHM = %s", "Age-Fitness")
        logging.info("  MUTATION RATE = %f", mutation_rate)
        logging.info("  CROSSOVER RATE = %f", crossover_rate)
        if req_x is not None:
            logging.info("  REQUIRED X (IMPLICIT REGRESSION) = %d", req_x)

def load_data(filename, step_size, fs_exponent):
    unformatted_data = []
    with h5py.File(filename,"r") as f: 
        hdf5_valueNames = list(f.keys())
        value_names = list()
        for exp in range(fs_exponent):
            for value in hdf5_valueNames:
                value_names.append(value + '^' + str(exp + 1))
                value_data = np.copy(f[value])
                value_data = np.power(value_data[::step_size], exp+1)
                unformatted_data.append(value_data)
    data = np.vstack([Udata for Udata in unformatted_data]).T
   
    if COMM_RANK == 0:
        LOGGER.info("DATA PARAMS:")
        LOGGER.info("  DATA FILE = %s", filename)
        LOGGER.info("  STEPPING THROUGH DATA AT STRIDES OF = %d", step_size)
        LOGGER.info("  DATA COLUMNS = [%s]", ", ".join(value_names))
        logging.info("")

    return data


def set_up_output_directory(dirname, checkpoint_name):
    if COMM_RANK == 0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    COMM.Barrier()
    os.chdir(dirname)

    if COMM_RANK == 0:
        LOGGER.info("OUTPUT PARAMS:")
        logging.info("  DIRECTORY = %s", dirname)
        logging.info("  CHECKPOINT NAME = %s", checkpoint_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--req_x", type=int,
                        help="number of x values required in implicit "
                             "regression")
    parser.add_argument("-k", "--data_step", type=int, default=1,
                        help="step size in stride through data")
    args = parser.parse_args()

    # INPUT DATA PARAMS
    FILENAME = "/uufs/chpc.utah.edu/common/home/u0871364/ML_project/midterm/circle_even1000_1.hdf5" 
    N_DATA_STEP = args.data_step
    FS_EXPONENT = 1  

    # OUTPUT PARAMS
    DIRNAME = 'pkl_files'
    CHECKPOINT_NAME = 'ML_project'

    # GP HYPER PARAMS
    N_GP_REPEATS = 20
    # convergence
    MAX_GENERATIONS = 100000
    ABS_TOLERANCE = 1e-20
    STAGNATION_GENERATIONS = 500000
    CONVERGENCE_CHECK_FREQ = 1000
    # equations
    EQU_POP_SIZE = 128
    EQU_SIZE = 100
    OPERATORS = ["+", "-", "*",'/']
    # fitness predictors
    PRED_POP_SIZE = 32
    PRED_SIZE_RATIO = 0.05
    PRED_COMP_RATIO = 0.1
    # variation
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.75

    set_up_output_directory(DIRNAME, CHECKPOINT_NAME)

    run_gp(FILENAME, N_DATA_STEP, N_GP_REPEATS, CHECKPOINT_NAME,
           MAX_GENERATIONS, ABS_TOLERANCE, STAGNATION_GENERATIONS,
           FS_EXPONENT, CONVERGENCE_CHECK_FREQ,
           EQU_POP_SIZE, EQU_SIZE, OPERATORS,
           PRED_POP_SIZE, PRED_SIZE_RATIO, PRED_COMP_RATIO,
           MUTATION_RATE, CROSSOVER_RATE, args.req_x)
    
