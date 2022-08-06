from typing import Dict
import collections
from datetime import datetime
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode = MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND


class MpiLogger:
    def __init__(self, maxlen):
        self.queue = collections.deque(maxlen=maxlen)

    def log_highload(self, filename, message):
        # now = datetime.now()
        # date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        # self.queue.append('[' + date_time + '] ' + message + '\n')
        # with open(filename, 'w') as f:
        #     for str in self.queue:
        #         f.write(str)
        pass

    def log2txt(self, filename, message):
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        m = '[' + date_time + '] ' + message + '\n'
        fh = MPI.File.Open(comm, filename, amode)
        fh.Write_all(m)
        fh.Close()

    def logDict(self, filename, d):
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        log_string = ''
        for key, value in d.items():
            log_string += "{0}: {1} ".format(key, value)
        m = '[' + date_time + '] ' + log_string + '\n'
        fh = MPI.File.Open(comm, filename, amode)
        fh.Write_all(m)
        fh.Close()
