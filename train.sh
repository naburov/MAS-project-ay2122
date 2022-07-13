# docker run -it --mount "type=bind,source=/c/Users/burov/Projects/mas-project-burov-ay2122/config.py,target=/app/mas-project-burov-ay2122/config.py" --moun
#t "type=bind,source=/c/Users/burov/Projects/mas-project-burov-ay2122/train.sh,target=/train.sh" -v "/c/Users/burov/Projects/mas-project-burov-ay2122/volume:/mnt/mas-project-burov-ay2122/"  mas-project-burov-ay2122

source activate opensim-rl
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun -np 4 --allow-run-as-root python -m mpi4py train_ddpg.py