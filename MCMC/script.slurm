#!/bin/bash
#SBATCH --job-name=MCMC
#SBATCH --output=MCMC.txt
#SBATCH --error=error_MCMC.txt
#SBATCH --ntasks=40
#SBATCH --nodes=1
#SBATCH --partition=generic

#cd $SLURM_SUBMIT_DIR

#module load gmp/6.2.1-gcc-5.3.0

#export LD_LIBRARY_PATH=/home/ricardogonzatto/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ricardogonzatto/nlopt-2.7.1/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ricardogonzatto/mpsolve-3.2.1/src/libmps/.libs/

export PYTHONPATH=$PYTHONPATH:/home/ricardogonzatto/Non_linear_non_gaussianity_CBPF_Project/MCMC/parallel_MCMC/


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ricardogonzatto/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ricardogonzatto/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/ricardogonzatto/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ricardogonzatto/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/home/ricardogonzatto/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "/home/ricardogonzatto/miniforge3/etc/profile.d/mamba.sh"
fi

# <<< conda initialize <<<

mamba activate polymv

mkdir -p chains
mpirun -np 10 cobaya-run test_run.yaml --resume --debug 
