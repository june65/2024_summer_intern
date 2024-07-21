import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
        seq_list_acc = []
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
                scene_acc = np.zeros((len(people_data), 2, self.seq_len))
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

                    ###PHYSIOGCN###
                    acc_scene_people = np.zeros(scene_people_xy.shape)
                    acc_scene_people[:, 1:] = rel_scene_people[:, 1:] - rel_scene_people[:, :-1]
                    ###PHYSIOGCN###

                    scene[N_index, :, time_front:time_end] = scene_people_xy
                    scene_rel[N_index, :, time_front:time_end] = rel_scene_people
                    scene_acc[N_index, :, time_front:time_end] = rel_scene_people
                    _non_linear_ped.append(poly_fit(scene_people_xy, pred_len, threshold))
                    
                    scene_loss_mask[N_index, time_front:time_end] = 1
                    N_index += 1
                
                if N_index > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(N_index)
                    loss_mask_list.append(scene_loss_mask[:N_index])
                    seq_list.append(scene[:N_index])
                    seq_list_rel.append(scene_rel[:N_index])
                    seq_list_acc.append(scene_rel[:N_index])
        
        self.num_seq = len(seq_list)
        #print(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_acc = np.concatenate(seq_list_acc, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_acc = torch.from_numpy(seq_list_acc[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_acc = torch.from_numpy(seq_list_acc[:, :, self.obs_len:]).type(torch.float)
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
            s_obs = torch.stack([self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :],self.obs_traj_acc[start:end, :]], dim=0).permute(0, 3, 1, 2)
            self.S_obs.append(s_obs.clone())
            s_trgt = torch.stack([self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],self.pred_traj_acc[start:end, :]], dim=0).permute(0, 3, 1, 2)
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
            self.obs_traj_acc[start:end, :], self.pred_traj_acc[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.S_obs[index], self.S_trgt[index]
        ]
        return out

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
test_args = parser.parse_args()

# Get arguments for training

checkpoint_dir = './checkpoint/' + test_args.tag + '/'

args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)
    
dataset_path = './datasets/' + args.dataset + '/'
model_path = checkpoint_dir + args.dataset + '_best.pth'

# Data preparation
test_dataset = Dataload(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = graph_tern(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
model = model.cuda()
model.load_state_dict(torch.load(model_path), strict=False)


def test(KSTEPS=20):
    model.eval()
    model.n_smpl = KSTEPS
    ade_refi_all = []
    fde_refi_all = []

    progressbar = tqdm(range(len(test_loader)))
    progressbar.set_description('Testing {}'.format(test_args.tag))

    for batch_idx, batch in enumerate(test_loader):
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs, pruning=4, clustering=True)

        # Calculate ADEs and FDEs for each refined trajectory
        V_trgt_abs = S_trgt[:, 0].squeeze(dim=0)
        temp = (V_refi - V_trgt_abs).norm(p=2, dim=-1)
        ADEs = temp.mean(dim=1).min(dim=0)[0]
        FDEs = temp[:, -1, :].min(dim=0)[0]
        ade_refi_all.extend(ADEs.tolist())
        fde_refi_all.extend(FDEs.tolist())

        progressbar.update(1)

    progressbar.close()

    ade_refi = sum(ade_refi_all) / len(ade_refi_all)
    fde_refi = sum(fde_refi_all) / len(fde_refi_all)
    return ade_refi, fde_refi


def main():
    ade_refi, fde_refi = [], []

    # Repeat the evaluation to reduce randomness
    repeat = 10
    for i in range(repeat):
        temp = test(KSTEPS=test_args.n_samples)
        ade_refi.append(temp[0])
        fde_refi.append(temp[1])

    ade_refi = np.mean(ade_refi)
    fde_refi = np.mean(fde_refi)

    result_lines = ["Evaluating model: {}".format(test_args.tag),
                    "Refined_ADE: {0:.8f}, Refined_FDE: {1:.8f}".format(ade_refi, fde_refi)]

    for line in result_lines:
        print(line)


if __name__ == "__main__":
    main()