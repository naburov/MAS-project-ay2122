FROM naburov/mas-project-burov-ay2122-base

WORKDIR "/app/mas-project-burov-ay2122"
COPY ./MyEnv.py "/app/mas-project-burov-ay2122"
COPY ./train_ddpg.py "/app/mas-project-burov-ay2122"
COPY ./config.py "/app/mas-project-burov-ay2122"
COPY ./Trainers "/app/mas-project-burov-ay2122/Trainers"
COPY ./train_dreamer.py "/app/mas-project-burov-ay2122"
COPY ./train_dreamer_sendrecv.py "/app/mas-project-burov-ay2122"
COPY ./train_dreamer_non_block.py "/app/mas-project-burov-ay2122"
COPY BaseImage/entrypoint.sh "/app/mas-project-burov-ay2122"

RUN chmod 755 /app/mas-project-burov-ay2122/entrypoint.sh
RUN mkdir /mnt/mas-project-burov-ay2122
RUN mkdir /mnt/mas-project-burov-ay2122/checkpoints
#ENTRYPOINT ["/app/mas-project-burov-ay2122/entrypoint.sh"]
#CMD ["/bin/bash", "-c", "source activate opensim-rl; export OMPI_MCA_btl_vader_single_copy_mechanism=none; mpirun -np 20 --allow-run-as-root python train_dreamer.py; export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64$LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH"]