class Logger:
    def log2txt(self, filename, message):
        with open(filename, 'a') as f:
            f.write(message + '\n')
