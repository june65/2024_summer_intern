from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

class SceneDataset(Dataset):
	def __init__(self, data, resize, obs_len=8, pred_len=12 ,skip=1, threshold=0.002, min_ped=1):
		self.max_people = 0
		
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.seq_len = obs_len + pred_len
		self.skip = skip
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""
		self.scene_list = self.split_trajectories_by_scene(data, resize, skip, threshold, min_ped)
		

	def __len__(self):
		return self.num_seq
	
	def __getitem__(self, index):
		start, end = self.seq_start_end[index]
		scene = self.scene_list[index]
		out = [
				self.obs_traj[start:end, :], self.pred_traj[start:end, :],
				self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
				self.non_linear_ped[start:end], self.loss_mask[start:end, :],
				self.S_obs[index], self.S_trgt[index]
			]
		return scene, out

	def split_trajectories_by_scene(self, data, resize, skip=1, threshold=0.002, min_ped=1):
		
		scene_list = []
		num_peds_in_seq = []
		seq_list = []
		seq_list_rel = []
		loss_mask_list = []
		non_linear_ped = []


		for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False), desc='Prepare Dataset'):
			meta_df = meta_df.head(1000)
			#########데이터 로딩 시간
			
			meta_df['x'] = meta_df['x'] * resize
			meta_df['y'] = meta_df['y'] * resize

			
			file_data = meta_df.to_numpy()
			file_data = np.delete(file_data, 4, axis=1)
			file_data = np.delete(file_data, 4, axis=1)
			file_data = file_data.astype(float)
			times = np.unique(file_data[:, 0]).tolist() 
			time_data = []
			for time in times:
				time_data.append(file_data[file_data[:, 0] == time, :])
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
					time_front = times.index(scene_people_xy[0, 0]) - i
					time_end =  times.index(scene_people_xy[-1, 0]) - i + 1
					if time_end - time_front != self.seq_len:
						continue
					scene_people_xy = np.transpose(scene_people_xy[:, 2:])
				
					rel_scene_people = np.zeros(scene_people_xy.shape)
					rel_scene_people[:, 1:] = scene_people_xy[:, 1:]- scene_people_xy[:, :-1]
					scene[N_index, :, time_front:time_end] = scene_people_xy
					scene_rel[N_index, :, time_front:time_end] = rel_scene_people
					_non_linear_ped.append(poly_fit(scene_people_xy, self.pred_len, threshold))
					
					scene_loss_mask[N_index, time_front:time_end] = 1
					N_index += 1
				
				if N_index > min_ped:
					non_linear_ped += _non_linear_ped
					num_peds_in_seq.append(N_index)
					scene_list.append(meta_df.iloc()[0:1].sceneId.item())
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

		for i in range(len(self.seq_start_end)):
			start, end = self.seq_start_end[i]
			#obs_traj
			s_obs = torch.stack([self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :]], dim=0).permute(0, 3, 1, 2)
			self.S_obs.append(s_obs.clone())
			s_trgt = torch.stack([self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :]], dim=0).permute(0, 3, 1, 2)
			self.S_trgt.append(s_trgt.clone())
			
		'''
		trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
		meta.append(meta_df)
		scene_list.append(meta_df.iloc()[0:1].sceneId.item())
		'''
		return scene_list
	
