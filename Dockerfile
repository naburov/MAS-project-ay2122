FROM naburov/mas-project-burov-ay2122-base

WORKDIR "/app/mas-project-burov-ay2122"
COPY ./MyEnv.py "/app/mas-project-burov-ay2122"
COPY ./train_ddpg.py "/app/mas-project-burov-ay2122"
COPY ./config.py "/app/mas-project-burov-ay2122"
COPY ./Trainers "/app/mas-project-burov-ay2122/Trainers"
COPY ./train_dreamer.py "/app/mas-project-burov-ay2122"

RUN mkdir /mnt/mas-project-burov-ay2122
RUN mkdir /mnt/mas-project-burov-ay2122/checkpoints