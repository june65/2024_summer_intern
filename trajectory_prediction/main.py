import numpy as np
import math
from pathlib import Path
import argparse


def Fileload(path):
    #text file load
    data = []
    with open(path, 'r') as file:
        content = file.read() 
        lines = content.split("\n")  
        for line in lines:
            if line.strip(): 
                try:
                    line = line.strip().split("\t")
                    line = [float(i) for i in line]
                    data.append(line)
                except: None
    return np.asarray(data)

class Dataload():
    def __init__(self, paths, obs_len=8, pred_len=8, skip=1):
        self.paths = paths
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip

        all_files = sorted(Path(self.paths).iterdir())
        all_files = [file for file in all_files if file.is_file()]

        for files_path in all_files:
            file_data = Fileload(files_path)
            times = np.unique(file_data[:, 0]).tolist()
            time_data = []
            for time in times:
                time_data.append(file_data[time == file_data[:, 0], :])
            num_time = int(math.ceil((len(times) - self.c + 1) / skip))
            print(time_data) 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='zara1', help='Dataset name(eth,hotel,univ,zara1,zara2)')
parser.add_argument('--skip', default='1')
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
args = parser.parse_args()


datasets_path = './datasets/' + args.dataset + '/'
train_dataset = Dataload(datasets_path + 'train/',  obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=args.skip)
