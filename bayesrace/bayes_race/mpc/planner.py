"""	Generate reference for MPC using a trajectory generator.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from bayes_race.utils import Spline2D


def ConstantSpeed(x0, v0, track, N, Ts, projidx, scale=0.9):
	"""	generate a reference trajectory of size 2x(N+1)
		first column is x0

		x0 		: current position (2x1)
		v0 		: current velocity (scaler)
		track	: see bayes_race.tracks, example Rectangular
		N 		: no of reference points, same as horizon in MPC
		Ts 		: sampling time in MPC
		projidx : hack required when raceline is longer than 1 lap

	"""
	# project x0 onto raceline
	raceline = track.raceline
	xy, idx = track.project_fast(x=x0[0], y=x0[1], raceline=raceline[:,projidx:projidx+10])
	projidx = idx+projidx

	# start ahead of the current position
	start = track.raceline[:,:projidx+2]

	xref = np.zeros([2,N+1])
	xref[:2,0] = x0

	# use splines to sample points based on max acceleration
	dist0 = np.sum(np.linalg.norm(np.diff(start), 2, axis=0))
	dist = dist0
	v = v0
	for idh in range(1,N+1):
		dist += scale*v*Ts
		dist = dist % track.spline.s[-1]
		xref[:2,idh] = track.spline.calc_position(dist)
		v = track.spline_v.calc(dist)
	return xref, projidx


def FromTrajectory(x0, x_traj, y_traj, N, Ts, projidx=0):
	"""	Generate reference from pre-computed trajectory.
	
	Args:
		x0: Current position (2x1) [x, y]
		x_traj: x-coordinates of trajectory (array)
		y_traj: y-coordinates of trajectory (array)
		N: Horizon length (number of reference points - 1)
		Ts: Sampling time
		projidx: Starting index in trajectory (for wrapping)
	
	Returns:
		xref: Reference trajectory (2x(N+1))
		projidx: Updated projection index
	"""
	# Project current position onto trajectory
	# Find nearest point in trajectory
	traj_points = np.array([x_traj, y_traj])
	
	# Search around projidx for nearest point
	search_window = 20
	start_idx = max(0, projidx - search_window)
	end_idx = min(len(x_traj), projidx + search_window)
	
	# Compute distances
	distances = np.linalg.norm(traj_points[:, start_idx:end_idx] - x0.reshape(2, 1), axis=0)
	nearest_local_idx = np.argmin(distances)
	nearest_idx = start_idx + nearest_local_idx
	
	# Update projidx
	projidx = nearest_idx
	
	# Extract reference points
	xref = np.zeros([2, N+1])
	xref[:, 0] = x0
	
	# Extract N points ahead (with wrapping)
	for idh in range(1, N+1):
		idx = (projidx + idh) % len(x_traj)
		xref[0, idh] = x_traj[idx]
		xref[1, idh] = y_traj[idx]
	
	return xref, projidx