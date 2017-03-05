# coding: utf-8

import os
from config import *


def get_last_weights_file():
    epoch_file_name = []
    for file_name in os.listdir(weights_dir):
        epoch = file_name.split('-')[-1]
        if epoch:
            epoch_file_name.append((epoch, file_name))
    return max(epoch_file_name)
