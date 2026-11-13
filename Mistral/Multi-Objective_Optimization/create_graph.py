import re
import yaml
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mergekit.config import MergeConfiguration

class CreateGraph:
    def __init__(self, pareto_df, directions, figure_dir):
        self.pareto_df = pareto_df
        self.directions = directions
        self.figure_dir = figure_dir
        
    def pareto(self):

        x = []
        y = []

        for _, t in self.pareto_df.iterrows():
            v0 = t["value_0"]
            v1 = t["value_1"]

            # if self.directions[0].name == "MAXIMIZE":
            #     v0 = -v0
            # if self.directions[1].name == "MAXIMIZE":
            #     v1 = -v1

            x.append(v0)
            y.append(v1)

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y)
        plt.xlabel("bit(minimize)", fontsize=16)
        plt.ylabel("acc(maximize)", fontsize=16)

        plt.savefig(f'./{self.figure_dir}/pareto_fronts.png')
        plt.close()