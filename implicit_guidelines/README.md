# implicit_guidelines
This folder contains input files and results related to developing training guidelines to improve computational performance of GPSR within the context of implicit regression. 

## Folders
There are 18 folders, each associated with a particular dataset fs# (corresponding to dataset X_#, see Equations 3 and 4), number of datapoints d#, and stack size e#. Within each folder, we have: 
- pkl_files: a folder that contains PKL files that are output from Bingo
- doit.slurm: SLURM job script that is submitted to CHPC
- extract_HOFfitness.sh: a shell script that extracts fitness values from the LOG files in the pkl_files folder
- fs#d#e#_ even_fitness.txt: a text file containing fitness values that are output from extract_HOFfitness.sh
- run_bingoMLproject.py: Python script that defines Bingo inputs such as dataset file, hyperparameters, etc. 
- slurm.out: SLURM output file that displays output equations 

## circle_even1000_1.hdf5
Dataset file containing 1000 evenly spaced points around the unit circle

## post_proc_fork.ipynb
Jupyter notebook that can read in PKL files that are output from Bingo to display equations corresponding to a particular generation, etc. 

## How to check implementation
To simply check this implementation, you can change to Jacob's home directory on CHPC and run the following commands: 
```
cd /uufs/chpc.utah.edu/common/home/u0871364/ML_project/implicit_guidelines     # change to directory
cd fs_d_e_even      # replace underscores with dataset, number of datapoints, and stack size you would like to check 
sbatch doit.slurm     # submit SLURM job script
vim slurm-####.out      # replace hashtags with batch job number. this will contain the best equations for each generation as it runs
```
