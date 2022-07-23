from typing import Dict


class Logger:
    def log2txt(self, filename, message):
        with open(filename, 'a') as f:
            f.write(message + '\n')

    def logDict(self, filename, d):
        log_string = ''
        for key, value in d.items():
            log_string += "{0}: {1} ".format(key, value)
        with open(filename, 'a') as f:
            f.write(log_string + '\n')
