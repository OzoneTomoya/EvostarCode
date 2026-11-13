import re
import yaml
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mergekit.config import MergeConfiguration

class CreateGraph:
    def __init__(self, study_trials, figure_dir):
        self.study_trials = study_trials
        self.figure_dir = figure_dir
        
    def optimization_histroy(self):
        # compute the cumulative maximum value of acc
        self.study_trials['acc_max'] = self.study_trials['value'].cummax()

        # create an optimization graph
        plt.figure(figsize=(6, 6))
        plt.plot(self.study_trials['number'], self.study_trials['value'],
                color='blue', label='acc', linewidth=1)
        plt.plot(self.study_trials['number'], self.study_trials['acc_max'],
                color='red', label='acc_max', linewidth=1)

        max_trial = self.study_trials['number'].max()

        x = np.arange(0, max_trial + 1, 500)

        y = np.arange(0.0, 0.75, 0.1)

        plt.title("objective function", fontsize=16)
        plt.xlabel('trial', fontsize=16)
        plt.ylabel('value', fontsize=16)
        plt.xticks(x, fontsize=16)
        plt.yticks(y, fontsize=16)
        plt.xlim(0, max_trial)
        plt.ylim(0.0, 0.75)
        plt.grid(axis='y', alpha=0.7)

        # save a fig
        plt.savefig(f'./{self.figure_dir}/optimization.png')

        plt.close()
    