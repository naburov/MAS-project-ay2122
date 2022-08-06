#!/bin/sh
source activate opensim-rl
export PATH=/opt/conda/envs/opensim-rl/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/openmpi/bin
export LD_LIBRARY_PATH=/opt/conda/envs/opensim-rl/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/openmpi/bin
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun -np 2 --allow-run-as-root python train_ddpg.py

