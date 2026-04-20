"""	Adaptive Ensemble MPC simulation on ETHZ track with Slippery Split environment.
	
	This script demonstrates the adaptive ensemble MPC that uses online weight
	estimation to adapt to variable friction conditions.
	
	Uses ETHZ track and Oracle reference (Feasibility Test).
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time as tm
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from bayes_race.params import ORCA
from bayes_race.models import Dynamic
from bayes_race.tracks import ETHZ
from bayes_race.mpc.planner import FromTrajectory
from bayes_race.mpc.nmpc_ensemble import setupNLPEnsemble
from bayes_race.mpc.estimator import EnsembleWeightEstimator
from bayes_race.utils.friction import get_friction


#####################################################################
# Configuration

SAVE_RESULTS = True
TRACK_CONS = False
PLOT_RESULTS = True

# MPC settings
SAMPLING_TIME = 0.02
HORIZON = 20

# --- TUNING FOR STABILITY ON ICE ---
# Q: [X_error, Y_error]
# We set X_error very low (0.05) so the solver doesn't panic if it falls behind.
# We set Y_error very high (10.0) to prioritize staying on the path.
COST_Q = np.diag([0.05, 10.0])

# P: Terminal cost (match Q usually)
COST_P = np.diag([0.05, 10.0])

# R: [Throttle_Rate, Steering_Rate]
# We set Steering_Rate very high (10.0) to prevent oscillation/loops on ice.
COST_R = np.diag([0.01, 10.0])

# Simulation settings
SIM_TIME = 15.0  # Increased to complete full track
MAX_STEPS = int(SIM_TIME / SAMPLING_TIME)

# Estimator settings
ESTIMATOR_BUFFER_SIZE = 10  # Number of recent steps for weight estimation
ESTIMATOR_UPDATE_FREQ = 1  # Update weights every N steps (1 = every step)


# File paths
# ORACLE FEASIBILITY TEST: Using friction-aware reference to isolate controller performance
# This proves the adaptive mechanism works when given a feasible reference trajectory
# The Baseline should still fail (wrong model) while Adaptive should track perfectly
ORACLE_REF_PATH = 'bayes_race/raceline/src/ethz_oracle_ref.npz'  # Friction-aware reference (slows before ice)
OUTPUT_PATH = 'bayes_race/data/ensemble_adaptive_ethz.npz'

if not TRACK_CONS:
	SUFFIX = 'NOCONS-'
else:
	SUFFIX = ''


#####################################################################
# Load track

print("=" * 60)
print("Phase 4: Adaptive Ensemble MPC (Stability Tuned)")
print("=" * 60)
print("\n1. Loading ETHZ track...")

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
print(f"   ✓ Track loaded: {TRACK_NAME}")
print(f"   Track bounds: x=[{track.x_raceline.min():.2f}, {track.x_raceline.max():.2f}], "
      f"y=[{track.y_raceline.min():.2f}, {track.y_raceline.max():.2f}]")

# Check if track crosses y=-0.5 for friction map
ice_waypoints = np.sum(track.y_raceline < -0.5)
asphalt_waypoints = np.sum(track.y_raceline >= -0.5)
print(f"   Waypoints in ice zone (y<-0.5, right in plot): {ice_waypoints}")
print(f"   Waypoints in asphalt zone (y>=-0.5, left in plot): {asphalt_waypoints}")


#####################################################################
# Load Oracle reference trajectory (Oracle Feasibility Test)
# This reference is friction-aware and slows down before the ice zone
# This isolates controller performance from planner limitations

print("\n2. Loading Oracle reference trajectory...")

if os.path.exists(ORACLE_REF_PATH):
	oracle_ref = np.load(ORACLE_REF_PATH)
	x_traj = oracle_ref['x']
	y_traj = oracle_ref['y']
	# We don't strictly enforce velocity tracking anymore due to low Q[0],
	# but the reference path provides the "safe" line.
	use_oracle_ref = True
	print(f"   ✓ Loaded Oracle reference (Points: {len(x_traj)})")
else:
	raise FileNotFoundError(f"Oracle reference not found at {ORACLE_REF_PATH}. Run Phase 1 generation first.")


#####################################################################
# Load vehicle parameters and create ensemble models

print("\n3. Creating ensemble models...")

params = ORCA(control='pwm')

# Store nominal Df, Dr for friction scaling
Df_nominal = params['Df']
Dr_nominal = params['Dr']
print(f"   Nominal tire parameters: Df={Df_nominal:.4f}, Dr={Dr_nominal:.4f}")

# Dry model: μ = 1.0 (Index 0)
model_dry = Dynamic(**params)
print(f"   ✓ Dry model (Index 0): Df={params['Df']:.4f}, Dr={params['Dr']:.4f}")

# Ice model: μ = 0.3 (Index 1)
params_ice = params.copy()
params_ice['Df'] = params['Df'] * 0.3
params_ice['Dr'] = params['Dr'] * 0.3
model_ice = Dynamic(**params_ice)
print(f"   ✓ Ice model (Index 1): Df={params_ice['Df']:.4f}, Dr={params_ice['Dr']:.4f}")

# Ensemble: [dry, ice]
models = [model_dry, model_ice]

# Plant Model: Variable friction (will be updated each step)
model_plant = Dynamic(**params)
print("   ✓ Plant model: Variable friction (will be updated)")


#####################################################################
# Setup Ensemble MPC controller

print("\n4. Setting up Ensemble MPC controller...")

Ts = SAMPLING_TIME
n_states = model_dry.n_states
n_inputs = model_dry.n_inputs
horizon = HORIZON

nlp = setupNLPEnsemble(
	horizon=horizon,
	Ts=Ts,
	Q=COST_Q,
	P=COST_P,
	R=COST_R,
	params=params,
	models=models,
	track=track,
	track_cons=TRACK_CONS
)
print("   ✓ Ensemble MPC controller initialized")


#####################################################################
# Setup Weight Estimator

print("\n5. Setting up weight estimator...")

estimator = EnsembleWeightEstimator(
	models=models,
	buffer_size=ESTIMATOR_BUFFER_SIZE,
	Ts=Ts
)
print(f"   ✓ Estimator initialized (buffer_size={ESTIMATOR_BUFFER_SIZE})")


#####################################################################
# Initialize simulation

print("\n6. Initializing simulation...")

n_steps = min(MAX_STEPS, int(SIM_TIME / Ts))
states = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts

# Tracking errors
lateral_errors = np.zeros(n_steps+1)
friction_values = np.zeros(n_steps+1)

weights_history = np.zeros([len(models), n_steps+1])

# Initialize state
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
states[:,0] = x_init

# Initial weights
mu_init = get_friction(x_init[0], x_init[1])
weights_history[:, 0] = np.array([1.0, 0.0]) if mu_init >= 0.5 else np.array([0.0, 1.0])
estimator.weights = weights_history[:, 0].copy()
estimator.update_buffer(x_init, np.zeros(n_inputs))


#####################################################################
# Main simulation loop

print("\n7. Running adaptive simulation...")
print("-" * 60)

projidx = 0
failure_detected = False

for idt in range(n_steps - HORIZON):
	
	# Get state
	uprev = inputs[:,idt-1] if idt > 0 else np.zeros(n_inputs)
	x0 = states[:,idt]
	
	# Update Plant Friction
	mu_current = get_friction(x0[0], x0[1])
	model_plant.Df = Df_nominal * mu_current
	model_plant.Dr = Dr_nominal * mu_current
	friction_values[idt] = mu_current
	
	# Update Estimator
	if idt > 0:
		estimator.update_buffer(states[:,idt], inputs[:,idt-1])
	
	if idt >= ESTIMATOR_BUFFER_SIZE:
		weights = estimator.estimate_weights()
	else:
		weights = weights_history[:, 0]
	
	weights_history[:, idt] = weights
	
	# Get Reference
	xref, projidx = FromTrajectory(x0[:2], x_traj, y_traj, HORIZON, Ts, projidx)
	
	# Solve MPC
	try:
		umpc, fval, xmpc = nlp.solve(x0, xref[:2,:], uprev, weights)
	except Exception as e:
		print(f"   Step {idt}: MPC solver error: {e}")
		umpc = np.zeros((n_inputs, HORIZON))
	
	# Sanity check
	if np.any(np.isnan(umpc)):
		umpc = np.zeros((n_inputs, HORIZON))

	inputs[:,idt] = umpc[:,0]
	
	# Sim Plant
	x_next, _ = model_plant.sim_continuous(states[:,idt], umpc[:,0].reshape(-1,1), [0, Ts])
	states[:,idt+1] = x_next[:,-1]
	
	# Compute Error
	xyproj, _ = track.project(x=states[0,idt+1], y=states[1,idt+1], raceline=track.raceline)
	lateral_errors[idt+1] = np.linalg.norm(states[:2, idt+1] - xyproj)
	
	# Logging
	if (idt+1) % 50 == 0:
		print(f"   Step {idt+1}: μ={mu_current:.2f}, w_ice={weights[1]:.2f}, Err={lateral_errors[idt+1]:.3f}m")
		if lateral_errors[idt+1] > 0.3:
			print(f"      ⚠ HIGH ERROR: {lateral_errors[idt+1]:.3f}m")


# Final Save
if SAVE_RESULTS:
	print(f"\n8. Saving results to {OUTPUT_PATH}...")
	os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
	np.savez(
		OUTPUT_PATH,
		time=time[:n_steps-HORIZON+1],
		states=states[:,:n_steps-HORIZON+1],
		inputs=inputs[:,:n_steps-HORIZON],
		friction=friction_values[:n_steps-HORIZON+1],
		lateral_errors=lateral_errors[:n_steps-HORIZON+1],
		weights=weights_history[:,:n_steps-HORIZON+1],
	)
	print("   ✓ Done.")
