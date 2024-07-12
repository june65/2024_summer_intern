
import os
import numpy as np
import pickle
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

def Fileload(path, seqs_used=None):
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
    data = np.asarray(data)

    if seqs_used is not None:
        filtered_data = []
        for seq in seqs_used:
            filtered_data.append(data[data[:, 0] == seq])
        data = np.concatenate(filtered_data, axis=0)

    return data

class Dataload(Dataset):
    def __init__(self, paths, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1):
        self.max_people = 0
        self.paths = paths
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.seqs_used = []
        self.seq_order = []
        self.pedestrian_ids = []

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
                current_ped_ids = []
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
                    current_ped_ids.append(people_id) 
                    N_index += 1
                
                if N_index > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(N_index)
                    loss_mask_list.append(scene_loss_mask[:N_index])
                    seq_list.append(scene[:N_index])
                    seq_list_rel.append(scene_rel[:N_index])
                    self.seqs_used.append(times[i])
                    self.seq_order.append(i // self.skip)
                    self.pedestrian_ids.append(current_ped_ids)
        
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

def save_gif(_path, predictions, ground_truths, seqs_used, pedestrian_used):

    data = Fileload(_path, seqs_used)
    seqs = np.unique(data[:, 0]).astype(int)
    ids = np.unique(np.hstack(pedestrian_used)).astype(int)
    
    ped_id_seq_num = [[30] * (len(seqs)+1) for _ in range(max(ids)+1)]
    for ped_id in ids:
        for idx_seq, seq in enumerate(seqs):
            for sorted_idx, sorted_id_ped in enumerate(pedestrian_used[idx_seq]):
                if sorted_id_ped == ped_id:
                    ped_id_seq_num[ped_id][idx_seq] = sorted_idx
                    break      

    colors = [cm.rainbow(random.random()) for _ in range(len(ids))]
    id_color_map = {id_: color for id_, color in zip(ids, colors)}

    fig, ax = plt.subplots()
    scatters = {id_: ax.scatter([], [], color=id_color_map[id_]) for id_ in ids}
    pred_lines = {id_: ax.plot([], [], linestyle='--', color=id_color_map[id_])[0] for id_ in ids}
    gt_lines = {id_: ax.plot([], [], linestyle='-', color=id_color_map[id_])[0] for id_ in ids}

    ax.set_xlim(np.min(data[:, 2]), np.max(data[:, 2]))
    ax.set_ylim(np.min(data[:, 3]), np.max(data[:, 3]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pedestrian Movement')

    history = {id_: [] for id_ in ids}
    pred_history = {id_: [] for id_ in ids}
    gt_history = {id_: [] for id_ in ids}

    def update(frame):
        pedestrian_used[frame] = [int(x) for x in pedestrian_used[frame]]
        for id_ in pedestrian_used[frame]:
            seq_data = data[data[:, 0] == seqs[frame]]
            ped_data = seq_data[seq_data[:, 1] == id_]
            if len(ped_data) > 0:
                history[id_].append(ped_data[:, 2:4])
                if len(history[id_]) > 8:
                    history[id_].pop(0)
                offsets = np.concatenate(history[id_], axis=0)
                scatters[id_].set_offsets(offsets)
                alphas = np.linspace(0, 0.3, len(history[id_]) - 1).tolist() + [0.9]
                scatters[id_].set_alpha(alphas) 
                pred_in = predictions[frame][:,ped_id_seq_num[id_][frame]]
                gt_in = ground_truths[frame][:,ped_id_seq_num[id_][frame]]
                pred_history[id_].append(pred_in)
                gt_history[id_].append(gt_in)
                pred_offsets = np.concatenate(pred_history[id_], axis=0)
                gt_offsets = np.concatenate(gt_history[id_], axis=0)
                pred_lines[id_].set_data(pred_offsets[:, 0], pred_offsets[:, 1])
                gt_lines[id_].set_data(gt_offsets[:, 0], gt_offsets[:, 1])
                
        for id_remove in (np.setdiff1d(ids, pedestrian_used[frame])):
            scatters[id_remove].set_offsets([[-100, -100]])
            pred_lines[id_remove].set_data(-100 * np.ones(12), -100 * np.ones(12))
            gt_lines[id_remove].set_data(-100 * np.ones(12), -100 * np.ones(12))

        ax.set_title(f'Pedestrian Movement: Sequence {int(seqs[frame])}')
        return list(scatters.values()) + list(pred_lines.values()) + list(gt_lines.values()) 
    ani = animation.FuncAnimation(fig, update, frames=len(seqs), interval=150, blit=True)
    gif_path = _path.replace('.txt', '.mp4').replace('/datasets/', '/visual_GraphTERN/')
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    ani.save(gif_path, writer=animation.FFMpegWriter(bitrate=10000))
    #plt.show()
    
checkpoint_dir = './checkpoint/' + test_args.tag + '/'

args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

dataset_path = './datasets/' + args.dataset + '/'
model_path = checkpoint_dir + args.dataset + '_best.pth'

test_dataset = Dataload(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def main():
    
    model = graph_tern(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                    seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path), strict=False)

    all_predictions = []
    all_ground_truths = []
    seq_order = test_dataset.seq_order 
    pedestrian_ids = test_dataset.pedestrian_ids
    for batch_idx, batch in enumerate(test_loader):
        
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]
        
        V_init, V_pred, V_refi, valid_mask = model(S_obs, pruning=4, clustering=True)
        V_refi_new =torch.mean(V_refi, dim=0)
        all_predictions.append(V_refi_new.cpu().detach().numpy())
        all_ground_truths.append(S_trgt[:, 1].squeeze(dim=0).cpu().detach().numpy())

    sorted_predictions = [x for _, x in sorted(zip(seq_order, all_predictions))]
    sorted_ground_truths = [x for _, x in sorted(zip(seq_order, all_ground_truths))]

    save_gif('./datasets/zara1/test/crowds_zara01.txt', sorted_predictions, sorted_ground_truths, test_dataset.seqs_used, test_dataset.pedestrian_ids)
    
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
