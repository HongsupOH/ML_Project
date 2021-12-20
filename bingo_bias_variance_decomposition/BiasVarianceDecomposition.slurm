#!/bin/csh

#SBATCH --time=4:00:00 # walltime, abbreviated by -t
#SBATCH -o bv-w1-%j.out-%N # name of the stdout
#SBATCH -e bv-w1-%j.err-%N # name of the stderr
#SBATCH --nodes=1
#SBATCH --ntasks=5 # number of MPI tasks, abbreviated by -n # additional information for allocated clusters 

#SBATCH --account=coe-np
#SBATCH --partition=coestudent-shared-np

setenv WORKDIR $PWD
setenv SCRDIR /scratch/notchpeak/serial/$USER/$SLURM_JOB_ID

module load gcc mpich

source /uufs/chpc.utah.edu/common/home/u6019587/.venv/bingovenv_v2/bin/activate.csh
setenv PYTHONPATH /uufs/chpc.utah.edu/common/home/u1164746/bingo:~/src/bingo_bandgap2

mpiexec -n $SLURM_NTASKS python ~/src/bingo_bias_variance_decomposition/tests.py -e ~/src/bingo_bias_variance_decomposition/experiment_files/bv_decomp.json





