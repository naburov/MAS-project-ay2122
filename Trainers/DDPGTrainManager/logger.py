from typing import Dict
import collections
from datetime import datetime


class Logger:
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
        with open(filename, 'a') as f:
            f.write('[' + date_time + '] ' + message + '\n')

    def logDict(self, filename, d):
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        log_string = ''
        for key, value in d.items():
            log_string += "{0}: {1} ".format(key, value)
        with open(filename, 'a') as f:
            f.write('[' + date_time + '] ' + log_string + '\n')
