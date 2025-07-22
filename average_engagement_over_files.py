from datetime import datetime, time, timedelta
import os
import time
import pandas as pd

datetimes : list[str] = []
path_file_1: str = "data/annotations/dyad_01/group.task engagement.helenrisack.annotation~"
path_file_2: str = "data/annotations/dyad_01/task engagement.group.carlosgonzalez.annotation~"


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    file_1 = pd.read_csv(path_file_1)
    file_2 = pd.read_csv(path_file_2)
    print(f"File 1 has {len(file_1)} lines and File 2 has {len(file_2)} lines")