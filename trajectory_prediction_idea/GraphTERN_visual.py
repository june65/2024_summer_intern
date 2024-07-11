
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import random
#GraphTERN
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import *
from tqdm import tqdm
from pathlib import Path

def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def Fileload(path):
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
                except:
                    None
    return np.asarray(data)

class Dataload(Dataset):
    def __init__(self, paths, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1):
        self.max_people = 0
        self.paths = paths
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip

        all_files = sorted(Path(self.paths).iterdir())
        all_files = [file for file in all_files if file.is_file()]

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        for files_path in all_files:
            file_data = Fileload(files_path)
            times = np.unique(file_data[:, 0]).tolist()
            time_data = []
            for time in times:
                time_data.append(file_data[time == file_data[:, 0], :])
            num_time = len(times) - self.seq_len + 1
            
            for i in range(0, num_time * self.skip + 1, skip):
                scene_data = np.concatenate(time_data[i:i+self.seq_len],axis=0)
                #scene_data (dim=2) i~i+seq_len 전체 데이터
                people_data = np.unique(scene_data[:, 1])
                self.max_people = max(self.max_people, len(people_data))

                scene = np.zeros((len(people_data), 2, self.seq_len))
                scene_rel = np.zeros((len(people_data), 2, self.seq_len))
                scene_loss_mask = np.zeros((len(people_data), self.seq_len))
                
                _non_linear_ped = []
                N_index = 0
                # N_index = num_peds_considered
                for _ , people_id in enumerate(people_data):
                    scene_people_xy = scene_data[scene_data[:, 1] == people_id, :]
                    #scene_data[:, 1] = each scene_people id
                    #scene_people = 특정 인물이 들어가 있는 scene_data
                    scene_people_xy = np.around(scene_people_xy, decimals=4)
                    time_front = times.index(scene_people_xy[0, 0]) - i
                    time_end =  times.index(scene_people_xy[-1, 0]) - i + 1
                    if time_end - time_front != self.seq_len:
                        continue
                    scene_people_xy = np.transpose(scene_people_xy[:, 2:])
                
                    rel_scene_people = np.zeros(scene_people_xy.shape)
                    rel_scene_people[:, 1:] = scene_people_xy[:, 1:] - scene_people_xy[:, :-1]
                    #rel_scene_people[:, 1:] = 특정 인물의 위치변환 (속도)
                    scene[N_index, :, time_front:time_end] = scene_people_xy
                    scene_rel[N_index, :, time_front:time_end] = rel_scene_people
                    _non_linear_ped.append(poly_fit(scene_people_xy, pred_len, threshold))
                    
                    scene_loss_mask[N_index, time_front:time_end] = 1
                    N_index += 1
                
                if N_index > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(N_index)
                    loss_mask_list.append(scene_loss_mask[:N_index])
                    seq_list.append(scene[:N_index])
                    seq_list_rel.append(scene_rel[:N_index])
        
        self.num_seq = len(seq_list)
        #print(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        self.S_obs = []
        self.S_trgt = []

        pbar = tqdm(total=len(self.seq_start_end))
        pbar.set_description('Processing {0} dataset {1}'.format(self.paths.split('/')[-3], self.paths.split('/')[-2]))

        for i in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[i]
            #obs_traj
            s_obs = torch.stack([self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :]], dim=0).permute(0, 3, 1, 2)
            self.S_obs.append(s_obs.clone())
            s_trgt = torch.stack([self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :]], dim=0).permute(0, 3, 1, 2)
            self.S_trgt.append(s_trgt.clone())
            pbar.update(1)
        pbar.close()
    
    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.S_obs[index], self.S_trgt[index]
        ]
        return out

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
test_args = parser.parse_args()


def save_gif(_path):

    data = Fileload(_path)

    seqs = np.unique(data[:, 0])
    ids = np.unique(data[:, 1])
    colors = [cm.rainbow(random.random()) for _ in range(len(ids))]
    id_color_map = {id_: color for id_, color in zip(ids, colors)}

    last_seq = {id_: max(data[data[:, 1] == id_][:, 0]) for id_ in ids}


    fig, ax = plt.subplots()
    scatters = {id_: ax.scatter([], [], color=id_color_map[id_]) for id_ in ids}

    ax.set_xlim(np.min(data[:, 2]), np.max(data[:, 2]))
    ax.set_ylim(np.min(data[:, 3]), np.max(data[:, 3]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pedestrian Movement')

    history = {id_: [] for id_ in ids}

    def update(frame):
        seq_data = data[data[:, 0] == seqs[frame]]
        for id_ in ids:
            if seqs[frame] <= last_seq[id_]:
                ped_data = seq_data[seq_data[:, 1] == id_]
                if len(ped_data) > 0:
                    history[id_].append(ped_data[:, 2:4])
                    if len(history[id_]) > 8:
                        history[id_].pop(0)
                    offsets = np.concatenate(history[id_], axis=0)
                    scatters[id_].set_offsets(offsets)
                    #scatters[id_].set_alpha(np.linspace(0.1, 0.9, len(history[id_])))
                
                    alphas = np.linspace(0, 0.3, len(history[id_]) - 1).tolist() + [0.9]
                    scatters[id_].set_alpha(alphas)    
            else:
                scatters[id_].set_offsets([[-10,0]]) 
        ax.set_title(f'Pedestrian Movement: Sequence {int(seqs[frame])}')
        return list(scatters.values())
    
    ani = animation.FuncAnimation(fig, update, frames=len(seqs), interval=150, blit=True)
    gif_path = _path.replace('.txt', '.mp4').replace('/datasets/', '/visual_GraphTERN/')
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    ani.save(gif_path, writer='ffmpeg')
    #plt.show()

def main():
    save_gif('./datasets/zara1/val/biwi_eth_val.txt')
    '''
    for root, dirs, files in os.walk('./datasets/'):
        #print(files)
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                save_gif(file_path)
    '''
if __name__ == "__main__":
    main()
