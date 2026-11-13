import re
import yaml
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mergekit.config import MergeConfiguration

class CreateGraph:
    def __init__(self, study_trials, figure_dir):
        self.study_trials = study_trials
        self.figure_dir = figure_dir
        
    def each_params(self):
        params = self.study_trials.columns

        # create a graph for each param
        for param in params:
            title = ''

            # skip 'number', set title
            if param == 'number':
                continue
            elif param == 'value':
                title = 'acc'
            else:
                title = param.replace('params_', '')
            
            # create a graph
            plt.figure(figsize=(5, 5))
            plt.plot(self.study_trials['number'], self.study_trials[param], color='blue', label=title, linewidth=0.5)
            plt.title(title)
            plt.xlabel('trial')
            plt.ylabel('value')
            if title == 'acc':
                plt.ylim(0.0, 1.0)

            # save a fig
            file_name = title.split('/')[-1]
            plt.savefig(f'./{self.figure_dir}/{file_name}.png')

            plt.close()

    def optimization_histroy(self):
        # compute the cumulative maximum value of acc
        self.study_trials['acc_max'] = self.study_trials['value'].cummax()

        # create an optimization graph
        plt.figure(figsize=(5, 5))
        plt.scatter(self.study_trials['number'], self.study_trials['value'], color='blue', label='acc', s=3)
        plt.plot(self.study_trials['number'], self.study_trials['acc_max'], color='red', label='acc_max', linewidth=1)
        plt.title('optimization')
        plt.xlabel('trial')
        plt.ylabel('value')
        plt.ylim(0.0, 1.0)
        plt.legend()

        # save a fig
        plt.savefig(f'./{self.figure_dir}/optimization.png')

        plt.close()
    
    def best_params(self):
        # get best trial
        best_trial_id = self.study_trials['value'].idxmax()

        param_name = r'params_layer(\d+)-bit'
        params = self.study_trials.columns
        layers = []
        bit_values = []

        for col in self.study_trials.columns:
            match = re.match(param_name, col)
            if match:
                layer_num = int(match.group(1))  
                layers.append(layer_num)
                bit_values.append(self.study_trials[col][best_trial_id])
        
        sorted_pairs = sorted(zip(layers, bit_values), key=lambda x: x[0])
        layers, bit_values = zip(*sorted_pairs)

        plt.figure(figsize=(5, 5))
        plt.plot(layers, bit_values, marker='o', color='blue')
        plt.title('best')
        plt.xlabel('layer')
        plt.ylabel('bit')
        plt.xticks(range(min(layers), max(layers) + 1, 4))
        plt.yticks(range(5, 17, 1))  
        plt.grid(True)
        plt.tight_layout()

        file_name = "bit"
        plt.savefig(f'./{self.figure_dir}/best_{file_name}.png')

        plt.close()