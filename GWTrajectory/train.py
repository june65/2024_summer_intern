import torch
import torch.nn as nn
from utils.image_utils import get_patch, image2world, create_gaussian_map, combine_gaussian_maps
import matplotlib.pyplot as plt

def train(model, train_loader, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat):
	"""
	Run training for one epoch

	:param model: torch model
	:param train_loader: torch dataloader
	:param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param e: epoch number
	:param params: dict of hyperparameters
	:param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
	:return: train_ADE, train_FDE, train_loss for one epoch
	"""
	
	train_loss = 0
	train_ADE = []
	train_FDE = []
	model.train()
	counter = 0
	# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
	for batch_idx , (scene, batch) in enumerate(train_loader):
		# Stop training after 25 batches to increase evaluation frequency
		if dataset_name == 'sdd' and obs_len == 8 and batch_idx > 25:
			break

		# Get scene image and apply semantic segmentation
		#if e < params['unfreeze']:  # before unfreeze only need to do semantic segmentation once
		model.eval()
		scene_image = train_images[scene[0]].to(device).unsqueeze(0)
		model.train()
		# inner loop, for each trajectory in the scene

		S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]
		
		trajectory = S_obs[0,0,:,:,:]
		trajectory = trajectory.permute(1, 0, 2).contiguous()
		trajectory_trgt = S_trgt[0,0,:,:,:]
		trajectory_trgt = trajectory_trgt.permute(1, 0, 2).contiguous()

		for i in range(0, len(trajectory), batch_size):
			#if e >= params['unfreeze']:
			#	scene_image = train_images[scene].to(device).unsqueeze(0)

			# Create Heatmaps for past and ground-truth future trajectories
			_, _, H, W = scene_image.shape  # image shape

			gaussian_maps = []
			for j in range(len(trajectory)):
				lengths =((trajectory[i,:,0] - trajectory[j,:,0])**2 + (trajectory[i,:,1] - trajectory[j,:,1])**2).sqrt()
				length = lengths.mean()
				if j == i:
					gaussian_maps.append([])
				elif length< 20:
					gaussian_maps.append([])
				observed = trajectory[j, :, :].reshape(-1, 2).cpu().numpy()
				gaussian_map = create_gaussian_map(observed, H, W)
				gaussian_maps.append(gaussian_map)
			combined_gaussian_maps = []	

			for k in range(obs_len):
				combined_gaussian_map = combine_gaussian_maps([gaussian_maps[j][k] for j in range(len(trajectory)) if len(gaussian_maps[j]) != 0], H, W)
				combined_gaussian_maps.append(combined_gaussian_map)

			#combined_gaussian_maps = combine_gaussian_maps(combined_gaussian_maps, H, W)
			
			combined_gaussian_maps = torch.tensor(combined_gaussian_maps).reshape([-1, obs_len, H, W])
			combined_gaussian_maps = combined_gaussian_maps.float()
			combined_gaussian_maps = combined_gaussian_maps.to(device)
				
			selected_gaussian = combined_gaussian_maps[0, 0].cpu().numpy()
			plt.imshow(selected_gaussian, cmap='hot', interpolation='nearest')
			plt.title(f'Gaussian Map')
			plt.colorbar()
			plt.show()
			
			observed = trajectory[i, :, :].reshape(-1, 2).cpu().numpy()
			observed_map = get_patch(input_template, observed, H, W)
			observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

			gt_future = trajectory_trgt[i].unsqueeze(0).to(device)
			gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
			gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])

			gt_waypoints = gt_future[:,params['waypoints']]
			gt_waypoint_map = get_patch(input_template, gt_waypoints.reshape(-1, 2).cpu().numpy(), H, W)
			gt_waypoint_map = torch.stack(gt_waypoint_map).reshape([-1, gt_waypoints.shape[1], H, W])

			feature_input = torch.cat([combined_gaussian_maps, observed_map], dim=1)
			# Forward pass
			# Calculate features
			features = model.pred_features(feature_input)

			# Predict goal and waypoint probability distribution
			pred_goal_map = model.pred_goal(features)
			goal_loss = criterion(pred_goal_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
			gt_waypoints_maps_downsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gt_waypoint_map) for i in range(1, len(features))]
			gt_waypoints_maps_downsampled = [gt_waypoint_map] + gt_waypoints_maps_downsampled

			# Predict trajectory distribution conditioned on goal and waypoints
			traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gt_waypoints_maps_downsampled)]
			pred_traj_map = model.pred_traj(traj_input)
			traj_loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Backprop
			loss = goal_loss + traj_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				train_loss += loss
				# Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
				pred_traj = model.softargmax(pred_traj_map)
				pred_goal = model.softargmax(pred_goal_map[:, -1:])

				train_ADE.append(((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
				train_FDE.append(((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))

	train_ADE = torch.cat(train_ADE).mean()
	train_FDE = torch.cat(train_FDE).mean()

	return train_ADE.item(), train_FDE.item(), train_loss.item()
