# bingo_bias_variance_decomposition
This folder contains input files and results related to implementing the BVD fitness function. 

## Folders
### checkpoints_explicit, logs_explicit
Contain PKL files and LOG files, respectively, that are output from Bingo for ordinary explicit regression

### checkpoints_st_##, logs_st_##
Contain PKL files and LOG files, respectively, that are output from Bingo for the BVD fitness function at a stack size of ##

### experiment_files
Contains JSON files that define hyperparameters and output folders

## BiasVarianceDecomposition.slurm
SLURM job script that is submitted to CHPC

## bias_variance_decomp.py
Implementation of the BVD fitness function

## helpGenData.py, helper_funcs.py
General Python files to help generate datasets and process Bingo results

## helper.py
Python file that generates the dataset used for this portion of the project 

## openpkl.py
Python file to open PKL files that are output from Bingo

## result_helper.py, result_helper_bvd.py
Python files to plot results for ordinary explicit regression and the BVD fitness function, respectively

## tests.py
Python script that defines Bingo inputs such as dataset file, type of regression, etc.

## How to check implementation
To simply check this implementation, you can run the following commands: 
```
cd /uufs/chpc.utah.edu/common/home/u1314063/src/bingo_bias_variance_decomposition      # change to Hongsup's directory
sbatch BiasVarianceDecomposition.slurm      # submit SLURM job script
tail logs_st_40/bvd_st_40_......log      # check the output of a particular job from a LOG file
```
