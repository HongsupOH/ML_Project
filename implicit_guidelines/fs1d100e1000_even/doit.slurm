#!/bin/csh
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u0871364@utah.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=coe-np
#SBATCH --partition=coestudent-shared-np

# set data and working directories
setenv WORKDIR $PWD
setenv SCRDIR /scratch/kingspeak/serial/$USER/$SLURM_JOB_ID
setenv MV2_ENABLE_AFFINITY 0 

mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR

# load appropriate modules
module purge
ml python/3.7.3
ml gcc mpich

setenv PYTHONPATH /uufs/chpc.utah.edu/common/home/u0871364/bingo
source /uufs/chpc.utah.edu/common/home/u6019587/.venv/bingovenv_v2/bin/activate.csh

# run the program
mpiexec -n $SLURM_NTASKS python run_bingoMLproject.py -k 10 
cp -r $SCRDIR/pkl_files $WORKDIR
