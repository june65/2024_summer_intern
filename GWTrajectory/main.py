import numpy as np
import math
import random
import os
import pickle
from pathlib import Path
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import *
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily


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

#data augmentor
def data_sampler(S_obs, S_trgt, batch=4, scale=True, stretch=False, flip=True, rotation=True, noise=False):
    
    aug_So, aug_Sg = [], []
    for i in range(batch):
        S_obs_t, S_tr_t = S_obs.clone(), S_trgt.clone()
        if scale:
            S_obs_t, S_tr_t = random_scale(S_obs_t, S_tr_t, min=0.8, max=1.2)
        if stretch:
            S_obs_t, S_tr_t = random_stretch(S_obs_t, S_tr_t, min=0.8, max=1.2)
        if flip:
            S_obs_t, S_tr_t = random_flip(S_obs_t, S_tr_t)
        if rotation:
            S_obs_t, S_tr_t = random_rotation(S_obs_t, S_tr_t)
        if noise:
            S_obs_t, S_tr_t = random_noise(S_obs_t, S_tr_t)

        aug_So.append(S_obs_t.squeeze(dim=0))
        aug_Sg.append(S_tr_t.squeeze(dim=0))

    S_obs = torch.stack(aug_So).detach()
    S_trgt = torch.stack(aug_Sg).detach()

    return S_obs, S_trgt

def random_scale(S_obs, S_trgt, min=0.8, max=1.2):
    
    scale = random.uniform(min, max)
    return S_obs * scale, S_trgt * scale

def random_stretch(S_obs, S_trgt, min=0.9, max=1.1):
    
    scale = [random.uniform(min, max), random.uniform(min, max)]
    scale = torch.tensor(scale).cuda()
    scale_a = torch.sqrt(scale[0] * scale[1])
    return S_obs * scale, S_trgt * scale

def random_flip(S_obs, S_trgt):
    
    flip = random.choice([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    flip = torch.tensor(flip).cuda()
    return S_obs * flip, S_trgt * flip

def random_rotation(S_obs, S_trgt):
    
    theta = random.uniform(-math.pi, math.pi)
    theta = (theta // (math.pi/2)) * (math.pi/2)

    r_mat = [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]]
    r = torch.tensor(r_mat, dtype=torch.float, requires_grad=False).cuda()

    S_obs = torch.einsum('rc,natvc->natvr', r, S_obs)
    S_trgt = torch.einsum('rc,natvc->natvr', r, S_trgt)
    return S_obs, S_trgt

def random_noise(S_obs, S_trgt, std=0.01):
    
    noise_obs = torch.randn_like(S_obs) * std
    noise_tr = torch.randn_like(S_trgt) * std
    return S_obs + noise_obs, S_trgt + noise_tr

def gaussian_mixture_loss(W_pred, S_trgt, n_stop):
    # NMV(C*K) -> NVM(C*K)
    W_pred = W_pred.transpose(1, 2).contiguous()

    temp = S_trgt.chunk(chunks=n_stop, dim=1)
    W_trgt_list = [i.mean(dim=1) for i in temp]
    W_pred_list = W_pred.chunk(chunks=n_stop, dim=-1)

    loss_list = []
    for i in range(n_stop):
        # NVMC
        W_pred_one = W_pred_list[i]
        W_trgt_one = W_trgt_list[i]
        mix = Categorical(torch.nn.functional.softmax(W_pred_one[:, :, :, 4], dim=-1))
        comp = Independent(Normal(W_pred_one[:, :, :, 0:2], W_pred_one[:, :, :, 2:4].exp()), 1)
        gmm = MixtureSameFamily(mix, comp)
        loss_list.append(-gmm.log_prob(W_trgt_one))

    loss = torch.cat(loss_list, dim=0)
    return loss.mean()

def mse_loss(S_pred, S_trgt, loss_mask, training=True):
    # NTVC
    loss = (S_pred - S_trgt).norm(p=2, dim=3) ** 2
    loss = loss.mean(dim=1) * loss_mask
    if training:
        loss[loss > 1] = 0
    return loss.mean()

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_epgcn', type=int, default=1, help='Number of EPGCN layers for endpoint prediction')
parser.add_argument('--n_epcnn', type=int, default=6, help='Number of EPCNN layers for endpoint prediction')
parser.add_argument('--n_trgcn', type=int, default=1, help='Number of TRGCN layers for trajectory refinement')
parser.add_argument('--n_trcnn', type=int, default=3, help='Number of TRCNN layers for trajectory refinement')
parser.add_argument('--n_ways', type=int, default=3, help='Number of control points for endpoint prediction')
parser.add_argument('--n_smpl', type=int, default=20, help='Number of samples for refine')
parser.add_argument('--kernel_size', type=int, default=3)

# Data parameters
parser.add_argument('--dataset', default='zara1', help='Dataset name(eth,hotel,univ,zara1,zara2)')
parser.add_argument('--skip', type=int, default=1)
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)

# Training parameters
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=512, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=128, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')


args = parser.parse_args()


datasets_path = './datasets/' + args.dataset + '/'
checkpoint_dir = './checkpoint/' + args.tag + '/'

train_dataset = Dataload(datasets_path + 'train/',  obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = Dataload(datasets_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = graph_tern(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'mingit _val_epoch': -1, 'min_val_loss': 1e10}



def train(epoch):
    global metrics, model
    model.train()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]
        
        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)

        V_init, V_pred, V_refi, valid_mask = model(S_obs, S_trgt)
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask)
        loss = r_loss + m_loss

        if torch.isnan(loss):
            pass
        else:
            loss.backward()
            loss_batch += loss.item()
            
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()


        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['train_loss'].append(loss_batch / loader_len)

def valid(epoch):
    global metrics, constant_metrics, model
    model.eval()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs)

        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask, training=False)
        loss = r_loss + m_loss

        loss_batch += loss.item()
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best.pth')


def main():
    for epoch in range(args.num_epochs):
        train(epoch)
        valid(epoch)

        if args.use_lrschd:
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'], constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()
