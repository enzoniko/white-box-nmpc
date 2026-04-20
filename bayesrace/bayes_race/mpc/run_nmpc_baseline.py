"""	Baseline MPC failure demonstration on ETHZ track with Slippery Split environment.
	
	This script demonstrates that a standard non-adaptive MPC fails when
	the plant experiences variable friction (ice zone) but the MPC model
	assumes constant dry friction.
	
	Uses ETHZ track and generates plots similar to plot_error_mpc_orca.py
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time as tm
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ
from bayes_race.mpc.planner import ConstantSpeed, FromTrajectory
from bayes_race.mpc.nmpc import setupNLP
from bayes_race.utils.friction import get_friction


#####################################################################
# Configuration

SAVE_RESULTS = True
TRACK_CONS = False
PLOT_RESULTS = True

# MPC settings
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

# Simulation settings
SIM_TIME = 15.0  # Increased to complete full track
MAX_STEPS = int(SIM_TIME / SAMPLING_TIME)

# File paths
NAIVE_REF_PATH = 'bayes_race/raceline/src/ethz_naive_ref.npz'  # ETHZ naive reference from Phase 1
OUTPUT_PATH = 'bayes_race/data/baseline_failure_ethz.npz'

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''


#####################################################################
# Load track

print("=" * 60)
print("Phase 2: Baseline MPC Failure on ETHZ Track")
print("=" * 60)
print("\n1. Loading ETHZ track...")

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
print(f"   ✓ Track loaded: {TRACK_NAME}")
print(f"   Track bounds: x=[{track.x_raceline.min():.2f}, {track.x_raceline.max():.2f}], "
      f"y=[{track.y_raceline.min():.2f}, {track.y_raceline.max():.2f}]")

# Check if track crosses y=-0.5 for friction map
# Ice zone: y < -0.5 (right side of plot, since plot x-axis is -y)
ice_waypoints = np.sum(track.y_raceline < -0.5)
asphalt_waypoints = np.sum(track.y_raceline >= -0.5)
print(f"   Waypoints in ice zone (y<-0.5, right in plot): {ice_waypoints}")
print(f"   Waypoints in asphalt zone (y>=-0.5, left in plot): {asphalt_waypoints}")


#####################################################################
# Load naive reference trajectory (if available, otherwise use track raceline)

print("\n2. Loading reference trajectory...")

use_naive_ref = False
if os.path.exists(NAIVE_REF_PATH):
	try:
		naive_ref = np.load(NAIVE_REF_PATH)
		x_traj = naive_ref['x']
		y_traj = naive_ref['y']
		v_traj = naive_ref['v']
		use_naive_ref = True
		print(f"   ✓ Loaded naive reference from {NAIVE_REF_PATH}")
		print(f"   Reference points: {len(x_traj)}")
	except:
		print(f"   ⚠ Could not load naive reference, using track raceline")
		x_traj = track.x_raceline
		y_traj = track.y_raceline
		v_traj = track.v_raceline if hasattr(track, 'v_raceline') else None
else:
	print(f"   ⚠ Naive reference not found at {NAIVE_REF_PATH}")
	print(f"   Using track raceline as reference")
	x_traj = track.x_raceline
	y_traj = track.y_raceline
	v_traj = track.v_raceline if hasattr(track, 'v_raceline') else None


#####################################################################
# Load vehicle parameters and create models

print("\n3. Creating model instances...")

params = ORCA(control='pwm')

# Store nominal Df, Dr for friction scaling
Df_nominal = params['Df']
Dr_nominal = params['Dr']
print(f"   Nominal tire parameters: Df={Df_nominal:.4f}, Dr={Dr_nominal:.4f}")

# MPC Model: Fixed dry parameters (μ=1.0)
# This model is used in CasADi NLP and never modified
model_mpc = Dynamic(**params)
print("   ✓ MPC model: Fixed dry parameters (μ=1.0)")

# Plant Model: Variable friction (will be updated each step)
# This model is used for true plant simulation
model_plant = Dynamic(**params)
print("   ✓ Plant model: Variable friction (will be updated)")


#####################################################################
# Setup MPC controller

print("\n4. Setting up MPC controller...")

Ts = SAMPLING_TIME
n_states = model_mpc.n_states
n_inputs = model_mpc.n_inputs
horizon = HORIZON

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_mpc, track, track_cons=TRACK_CONS)
print("   ✓ MPC controller initialized with fixed dry model")


#####################################################################
# Initialize simulation

print("\n5. Initializing simulation...")

n_steps = min(MAX_STEPS, int(SIM_TIME / Ts))
states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts

# MPC predictions (what MPC thinks will happen with dry model)
mpc_predictions = np.zeros([n_states, horizon+1, n_steps])

# Plant predictions (what actually happens with variable friction)
plant_predictions = np.zeros([n_states, horizon+1, n_steps])

# Tracking errors
lateral_errors = np.zeros(n_steps+1)
friction_values = np.zeros(n_steps+1)

# Initialize state
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
states[:,0] = x_init
dstates[0,0] = x_init[3]
friction_values[0] = get_friction(x_init[0], x_init[1])

print(f"   Starting at ({x_init[0]:.2f}, {x_init[1]:.2f})")
print(f"   Simulation steps: {n_steps}")


#####################################################################
# Main simulation loop

print("\n6. Running simulation...")
print("-" * 60)

projidx = 0
failure_detected = False
failure_step = None

for idt in range(n_steps - horizon):
	
	# Get current state
	uprev = inputs[:,idt-1] if idt > 0 else np.zeros(n_inputs)
	x0 = states[:,idt]
	x_pos, y_pos = x0[0], x0[1]
	
	# Update plant model friction based on current position (y-coordinate determines friction)
	mu_current = get_friction(x_pos, y_pos)
	model_plant.Df = Df_nominal * mu_current
	model_plant.Dr = Dr_nominal * mu_current
	friction_values[idt] = mu_current
	
	# Get reference
	if use_naive_ref:
		# Use pre-computed naive reference trajectory
		xref, projidx = FromTrajectory(
			x0=x0[:2], 
			x_traj=x_traj, 
			y_traj=y_traj, 
			N=horizon, 
			Ts=Ts, 
			projidx=projidx
		)
	else:
		# Use ConstantSpeed planner with track raceline
		xref, projidx = ConstantSpeed(
			x0=x0[:2], 
			v0=x0[3], 
			track=track, 
			N=horizon, 
			Ts=Ts, 
			projidx=projidx
		)
	
	# Solve MPC (uses fixed dry model)
	try:
		start = tm.time()
		umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)
		solve_time = tm.time() - start
		
		# Check for solver failure
		if fval > 1e6 or np.any(np.isnan(umpc)) or np.any(np.isinf(umpc)):
			print(f"   Step {idt}: MPC solver failed (cost={fval:.2e})")
			if idt > 0:
				umpc = np.tile(inputs[:,idt-1].reshape(-1, 1), (1, horizon))
			else:
				umpc = np.zeros((n_inputs, horizon))
			fval = 1e6
			xmpc = np.zeros((n_states, horizon+1))
	except Exception as e:
		print(f"   Step {idt}: MPC solver error: {e}")
		if idt > 0:
			umpc = np.tile(inputs[:,idt-1].reshape(-1, 1), (1, horizon))
		else:
			umpc = np.zeros((n_inputs, horizon))
		fval = 1e6
		xmpc = np.zeros((n_states, horizon+1))
	
	inputs[:,idt] = umpc[:,0]
	
	# Simulate plant (uses variable friction model)
	x_next, dxdt_next = model_plant.sim_continuous(
		states[:,idt], 
		umpc[:,0].reshape(-1,1), 
		[0, Ts]
	)
	states[:,idt+1] = x_next[:,-1]
	dstates[:,idt+1] = dxdt_next[:,-1]
	friction_values[idt+1] = get_friction(states[0,idt+1], states[1,idt+1])
	
	# Compute tracking error (project onto raceline)
	xyproj, _ = track.project(x=states[0,idt+1], y=states[1,idt+1], raceline=track.raceline)
	ref_point = xyproj
	actual_point = states[:2, idt+1]
	lateral_errors[idt+1] = np.linalg.norm(actual_point - ref_point)
	
	# Forward simulate MPC prediction (dry model)
	mpc_pred = np.zeros((n_states, horizon+1))
	mpc_pred[:,0] = x0
	for idh in range(horizon):
		x_pred, _ = model_mpc.sim_continuous(
			mpc_pred[:,idh], 
			umpc[:,idh].reshape(-1,1), 
			[0, Ts]
		)
		mpc_pred[:,idh+1] = x_pred[:,-1]
	mpc_predictions[:,:,idt] = mpc_pred
	
	# Forward simulate plant prediction (variable friction)
	plant_pred = np.zeros((n_states, horizon+1))
	plant_pred[:,0] = x0
	for idh in range(horizon):
		# Update friction for prediction
		pred_pos_x, pred_pos_y = plant_pred[0,idh], plant_pred[1,idh]
		pred_mu = get_friction(pred_pos_x, pred_pos_y)
		model_plant.Df = Df_nominal * pred_mu
		model_plant.Dr = Dr_nominal * pred_mu
		
		x_pred, _ = model_plant.sim_continuous(
			plant_pred[:,idh], 
			umpc[:,idh].reshape(-1,1), 
			[0, Ts]
		)
		plant_pred[:,idh+1] = x_pred[:,-1]
	plant_predictions[:,:,idt] = plant_pred
	
	# Check for failure (large tracking error or out of bounds)
	if lateral_errors[idt+1] > 0.2:  # 20cm error threshold for ETHZ track
		if not failure_detected:
			failure_detected = True
			failure_step = idt + 1
			print(f"\n   ⚠ FAILURE DETECTED at step {failure_step} (t={time[failure_step]:.2f}s)")
			print(f"      Position: ({x_pos:.2f}, {y_pos:.2f})")
			print(f"      Friction: μ={mu_current:.2f}")
			print(f"      Tracking error: {lateral_errors[idt+1]:.3f} m")
	
	# Print progress
	if (idt + 1) % 50 == 0 or failure_detected:
		print(f"   Step {idt+1}/{n_steps-horizon}: t={time[idt+1]:.2f}s, "
		      f"pos=({x_pos:.2f}, {y_pos:.2f}), μ={mu_current:.2f}, "
		      f"error={lateral_errors[idt+1]:.3f}m, cost={fval:.3f}")


#####################################################################
# Finalize results

print("\n" + "=" * 60)
print("Simulation Complete")
print("=" * 60)

if failure_detected:
	print(f"✓ Failure detected at step {failure_step} (t={time[failure_step]:.2f}s)")
	print(f"  Final tracking error: {lateral_errors[n_steps-horizon]:.3f} m")
	print(f"  Max tracking error: {lateral_errors[:n_steps-horizon+1].max():.3f} m")
else:
	print("⚠ No explicit failure detected, but check results for deviations")

# Compute statistics
ice_steps = np.where(friction_values[:n_steps-horizon+1] < 0.5)[0]
asphalt_steps = np.where(friction_values[:n_steps-horizon+1] >= 0.5)[0]

if len(ice_steps) > 0:
	print(f"\nStatistics:")
	print(f"  Steps in ice zone: {len(ice_steps)}")
	print(f"  Mean error in ice: {lateral_errors[ice_steps].mean():.3f} m")
	print(f"  Max error in ice: {lateral_errors[ice_steps].max():.3f} m")

if len(asphalt_steps) > 0:
	print(f"  Steps in asphalt: {len(asphalt_steps)}")
	print(f"  Mean error in asphalt: {lateral_errors[asphalt_steps].mean():.3f} m")
	print(f"  Max error in asphalt: {lateral_errors[asphalt_steps].max():.3f} m")


#####################################################################
# Save results

if SAVE_RESULTS:
	print(f"\n7. Saving results to {OUTPUT_PATH}...")
	os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
	
	np.savez(
		OUTPUT_PATH,
		time=time[:n_steps-horizon+1],
		states=states[:,:n_steps-horizon+1],
		dstates=dstates[:,:n_steps-horizon+1],
		inputs=inputs[:,:n_steps-horizon],
		friction=friction_values[:n_steps-horizon+1],
		lateral_errors=lateral_errors[:n_steps-horizon+1],
		mpc_predictions=mpc_predictions[:,:,:n_steps-horizon],
		plant_predictions=plant_predictions[:,:,:n_steps-horizon],
		failure_detected=failure_detected,
		failure_step=failure_step,
		)
	print(f"   ✓ Results saved")


#####################################################################
# Generate plots (similar to plot_error_mpc_orca.py)

if PLOT_RESULTS:
	print("\n8. Generating plots...")
	
	# Plot 1: Error plot similar to plot_error_mpc_orca.py
	plt.figure(figsize=(6, 4))
	plt.axis('equal')
	
	# Plot track boundaries
	plt.plot(-track.y_outer, track.x_outer, 'k', lw=0.5, alpha=0.5)
	plt.plot(-track.y_inner, track.x_inner, 'k', lw=0.5, alpha=0.5)
	
	# Plot actual trajectory (ground truth)
	plt.plot(-states[1,:n_steps-horizon+1], states[0,:n_steps-horizon+1], '-k', lw=0.5, label='actual')
	
	# Plot predictions at various time steps (similar to plot_error_mpc_orca.py)
	SAMPLING_TIME_PLOT = SAMPLING_TIME
	# Remove hardcoded limit to show full simulation
	index_range = range(HORIZON+5, n_steps-horizon, HORIZON+5)
	
	INDEX = 0
	if INDEX < n_steps-horizon:
		# Plant prediction (what actually happens with variable friction)
		plant_pred = plant_predictions[:,:,INDEX]
		plt.plot(-plant_pred[1,:], plant_pred[0,:], '-g', marker='o', markersize=1, lw=0.5, 
		        label='simulated ($f_{\mathrm{plant}}$)')
		# MPC prediction (what MPC thinks will happen with dry model)
		mpc_pred = mpc_predictions[:,:,INDEX]
		plt.plot(-mpc_pred[1,:], mpc_pred[0,:], '-r', marker='o', markersize=1, lw=0.5, 
		        label='predicted ($f_{\mathrm{mpc}}$)')
		plt.scatter(-states[1,INDEX], states[0,INDEX], color='b', marker='o', alpha=0.8, s=15)
		plt.text(-states[1,INDEX], states[0,INDEX]+0.05, '0', color='k', fontsize=10, ha='center', va='bottom')
	
	for INDEX in index_range:
		if INDEX < n_steps-horizon:
			plant_pred = plant_predictions[:,:,INDEX]
			mpc_pred = mpc_predictions[:,:,INDEX]
			plt.plot(-plant_pred[1,:], plant_pred[0,:], '-g', marker='o', markersize=1, lw=0.5)
			plt.plot(-mpc_pred[1,:], mpc_pred[0,:], '-r', marker='o', markersize=1, lw=0.5)
			plt.scatter(-states[1,INDEX], states[0,INDEX], color='b', marker='o', alpha=0.8, s=15)
			plt.text(-states[1,INDEX]+0.05, states[0,INDEX]+0.05, '{:.1f}'.format(INDEX*SAMPLING_TIME_PLOT), 
			        color='k', fontsize=10)
	
	plt.xlabel('$x$ [m]')
	plt.ylabel('$y$ [m]')
	plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.15), frameon=False)
	plt.savefig('bayes_race/mpc/error_baseline_mpc.png', dpi=600, bbox_inches='tight')
	print("   ✓ Error plot saved (error_baseline_mpc.png)")
	plt.close()
	
	# Plot 2: Trajectory comparison with friction zones
	fig, ax = plt.subplots(figsize=(10, 8))
	ax.set_aspect('equal')
	
	# Track boundaries
	ax.plot(-track.y_outer, track.x_outer, 'k', lw=1, alpha=0.3)
	ax.plot(-track.y_inner, track.x_inner, 'k', lw=1, alpha=0.3)
	
	# Reference raceline
	ax.plot(-track.y_raceline, track.x_raceline, '--b', linewidth=1.5, alpha=0.5, label='Reference')
	
	# Actual trajectory
	ax.plot(-states[1,:n_steps-horizon+1], states[0,:n_steps-horizon+1], 'r-', linewidth=2, label='Actual (Plant)')
	
	# Highlight ice zone on track based on actual friction map
	# Friction map: ice when y < -0.5 (right side of plot, since plot x-axis is -y)
	ice_mask_track = track.y_raceline < -0.5
	if np.any(ice_mask_track):
		ax.scatter(-track.y_raceline[ice_mask_track], track.x_raceline[ice_mask_track], 
		          c='cyan', s=10, alpha=0.3, label='Ice Zone (y < -0.5, right in plot)', zorder=0)
	
	# Also highlight actual trajectory points in ice zone
	# Use actual friction values from simulation to determine ice zone
	traj_indices = np.arange(n_steps-horizon+1)
	ice_mask_traj = friction_values[:n_steps-horizon+1] < 0.5
	if np.any(ice_mask_traj):
		ice_indices = traj_indices[ice_mask_traj]
		ax.scatter(-states[1,ice_indices], states[0,ice_indices], 
		          c='cyan', s=5, alpha=0.5, marker='x', zorder=1)
	
	# Show predictions at key points
	key_steps = [n_steps//4, n_steps//2, 3*n_steps//4]
	if failure_step is not None:
		key_steps.append(min(failure_step, n_steps-horizon-1))
	key_steps = sorted(set([s for s in key_steps if s < n_steps-horizon]))[:4]
	
	for step in key_steps:
		mpc_pred = mpc_predictions[:,:,step]
		plant_pred = plant_predictions[:,:,step]
		ax.plot(-mpc_pred[1,:], mpc_pred[0,:], 'g--', linewidth=1.5, alpha=0.7, 
		       marker='o', markersize=3, label='MPC (Dry)' if step == key_steps[0] else '')
		ax.plot(-plant_pred[1,:], plant_pred[0,:], 'm--', linewidth=1.5, alpha=0.7,
		       marker='s', markersize=3, label='Plant (Variable μ)' if step == key_steps[0] else '')
	
	ax.set_xlabel('$x$ [m]')
	ax.set_ylabel('$y$ [m]')
	ax.set_title('Baseline MPC Failure: ETHZ Track')
	ax.legend(loc='best')
	ax.grid(True, alpha=0.3)
	plt.savefig('bayes_race/mpc/baseline_failure_ethz_trajectory.png', dpi=150, bbox_inches='tight')
	print("   ✓ Trajectory plot saved (baseline_failure_ethz_trajectory.png)")
	plt.close()
	
	# Plot 3: Tracking error and friction over time
	fig, axes = plt.subplots(2, 1, figsize=(10, 8))
	
	# Error over time
	ax1 = axes[0]
	ax1.plot(time[:n_steps-horizon+1], lateral_errors[:n_steps-horizon+1], 'r-', linewidth=2)
	ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Failure threshold')
	if failure_step is not None:
		ax1.axvline(x=time[failure_step], color='red', linestyle=':', 
		           linewidth=2, label='Failure detected')
	ax1.set_ylabel('Tracking Error [m]')
	ax1.set_title('Lateral Tracking Error')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	# Friction over time
	ax2 = axes[1]
	ax2.plot(time[:n_steps-horizon+1], friction_values[:n_steps-horizon+1], 'b-', linewidth=2)
	ax2.axhline(y=0.3, color='cyan', linestyle='--', alpha=0.5, label='Ice (μ=0.3, y<-0.5)')
	ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Asphalt (μ=1.0, y>=-0.5)')
	ax2.set_ylabel('Friction Coefficient μ')
	ax2.set_xlabel('Time [s]')
	ax2.set_title('Friction Map (Ice: y<-0.5, right in plot)')
	ax2.legend()
	ax2.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig('bayes_race/mpc/baseline_failure_ethz_analysis.png', dpi=150, bbox_inches='tight')
	print("   ✓ Analysis plot saved (baseline_failure_ethz_analysis.png)")
	plt.close()

print("\n" + "=" * 60)
print("Phase 2 Complete!")
print("=" * 60)
