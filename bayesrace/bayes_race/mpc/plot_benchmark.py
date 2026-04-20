"""	Plot benchmark comparison: Oracle, Baseline, and Ensemble trajectories.
	
	Generates publication-quality figures comparing:
	1. Trajectory comparison on track map
	2. Velocity and lateral error over time
	3. Weight adaptation over time
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_race.tracks import ETHZ
from bayes_race.utils.friction import get_friction


#####################################################################
# Configuration

SAVE_RESULTS = True
HORIZON = 20  # For prediction visualization
SAMPLING_TIME = 0.02

# File paths
ORACLE_REF_PATH = 'bayes_race/raceline/src/ethz_oracle_ref.npz'
NAIVE_REF_PATH = 'bayes_race/raceline/src/ethz_naive_ref.npz'
BASELINE_DATA_PATH = 'bayes_race/data/baseline_failure_ethz.npz'
ENSEMBLE_DATA_PATH = 'bayes_race/data/ensemble_adaptive_ethz.npz'

# Output paths
FIG1_PATH = 'bayes_race/mpc/benchmark_trajectories.png'
FIG2_PATH = 'bayes_race/mpc/benchmark_states_errors.png'
FIG3_PATH = 'bayes_race/mpc/benchmark_adaptation.png'


#####################################################################
# Load track

print("=" * 60)
print("Phase 4: Benchmark Comparison Plotting")
print("=" * 60)
print("\n1. Loading track...")

track = ETHZ(reference='optimal', longer=True)
print(f"   ✓ Track loaded")


#####################################################################
# Load data

print("\n2. Loading simulation data...")

# Oracle reference (from Phase 1)
oracle_ref = np.load(ORACLE_REF_PATH)
x_oracle = oracle_ref['x']
y_oracle = oracle_ref['y']
v_oracle = oracle_ref['v']
t_oracle = oracle_ref['t']
print(f"   ✓ Oracle reference loaded: {len(x_oracle)} points, lap time={t_oracle:.2f}s")

# Naive reference (from Phase 1)
naive_ref = np.load(NAIVE_REF_PATH)
x_naive = naive_ref['x']
y_naive = naive_ref['y']
v_naive = naive_ref['v']
t_naive = naive_ref['t']
print(f"   ✓ Naive reference loaded: {len(x_naive)} points, lap time={t_naive:.2f}s")

# Baseline data (from Phase 2)
baseline_data = np.load(BASELINE_DATA_PATH)
time_baseline = baseline_data['time']
states_baseline = baseline_data['states']
lateral_errors_baseline = baseline_data['lateral_errors']
friction_baseline = baseline_data['friction']
print(f"   ✓ Baseline data loaded: {len(time_baseline)} steps")

# Ensemble data (from Phase 4)
ensemble_data = np.load(ENSEMBLE_DATA_PATH)
time_ensemble = ensemble_data['time']
states_ensemble = ensemble_data['states']
lateral_errors_ensemble = ensemble_data['lateral_errors']
friction_ensemble = ensemble_data['friction']
weights_ensemble = ensemble_data['weights']  # (n_models, n_steps)
ensemble_predictions = ensemble_data.get('ensemble_predictions', None)  # Optional
print(f"   ✓ Ensemble data loaded: {len(time_ensemble)} steps")


#####################################################################
# Figure 1: Trajectory Comparison

print("\n3. Generating Figure 1: Trajectory Comparison...")

fig1, ax1 = plt.subplots(figsize=(12, 10))
ax1.set_aspect('equal')

# Plot track boundaries
ax1.plot(-track.y_outer, track.x_outer, 'k', lw=1, alpha=0.3, label='Track boundaries')
ax1.plot(-track.y_inner, track.x_inner, 'k', lw=1, alpha=0.3)

# Highlight ice zone (y < -0.5, appears as right side in plot since x-axis is -y)
ice_mask_track = track.y_raceline < -0.5
if np.any(ice_mask_track):
	ax1.scatter(-track.y_raceline[ice_mask_track], track.x_raceline[ice_mask_track], 
	          c='cyan', s=15, alpha=0.2, label='Ice Zone (y < -0.5, right in plot)', 
	          zorder=0, edgecolors='none')

# Plot Oracle reference (optimal trajectory with friction awareness)
# Note: Oracle and Naive have same PATH (x,y coordinates) but different VELOCITIES
# Oracle slows down in ice zone, Naive maintains high speed
# The path alignment is expected - the difference is in speed profile, shown in velocity plot
ax1.plot(-y_oracle, x_oracle, 'b-', linewidth=2.5, alpha=0.8, 
        label='Oracle Reference (optimal with friction map, slower in ice)', zorder=2)

# Plot Naive reference (what MPC is tracking - assumes dry everywhere)
# This has the same path as Oracle but maintains high speed throughout
ax1.plot(-y_naive, x_naive, 'g--', linewidth=2, alpha=0.6, 
        label='Naive Reference (assumes μ=1.0 everywhere, same path, higher speed)', zorder=2)

# Plot Baseline trajectory
ax1.plot(-states_baseline[1, :], states_baseline[0, :], 'r-', linewidth=2.5, 
        label='Baseline MPC (fixed dry model)', zorder=3)

# Plot Ensemble trajectory
ax1.plot(-states_ensemble[1, :], states_ensemble[0, :], 'm-', linewidth=2.5, 
        label='Ensemble MPC (adaptive)', zorder=3)

# Add time markings and predictions (similar to error_baseline_mpc.png)
# Show predictions at various time steps throughout the simulation
# Use same spacing as error_baseline_mpc.png: every HORIZON+5 steps
n_ensemble_steps = len(time_ensemble) - HORIZON
n_baseline_steps = len(time_baseline) - HORIZON
index_range_ensemble = list(range(HORIZON+5, n_ensemble_steps, HORIZON+5))
index_range_baseline = list(range(HORIZON+5, n_baseline_steps, HORIZON+5))
# Show more predictions throughout the simulation (not just first 5)
max_predictions = min(15, len(index_range_ensemble))  # Show up to 15 prediction sets

# Baseline predictions (if available)
baseline_predictions = baseline_data.get('mpc_predictions', None)
if baseline_predictions is not None:
	INDEX = 0
	if INDEX < n_baseline_steps:
		baseline_pred = baseline_predictions[:,:,INDEX]
		ax1.plot(-baseline_pred[1,:], baseline_pred[0,:], '-r', marker='o', 
		        markersize=1, lw=0.5, alpha=0.5, label='Baseline predictions' if INDEX == 0 else '')
	
	# Show predictions throughout simulation
	for idx, INDEX in enumerate(index_range_baseline[:max_predictions]):
		if INDEX < n_baseline_steps:
			baseline_pred = baseline_predictions[:,:,INDEX]
			ax1.plot(-baseline_pred[1,:], baseline_pred[0,:], '-r', marker='o', 
			        markersize=1, lw=0.5, alpha=0.5)
			ax1.scatter(-states_baseline[1,INDEX], states_baseline[0,INDEX], 
			          color='r', marker='o', alpha=0.6, s=15)
			ax1.text(-states_baseline[1,INDEX]+0.05, states_baseline[0,INDEX]+0.05, 
			        '{:.1f}'.format(time_baseline[INDEX]), color='r', fontsize=9)

# Ensemble predictions
if ensemble_predictions is not None:
	INDEX = 0
	if INDEX < n_ensemble_steps:
		ensemble_pred = ensemble_predictions[:,:,INDEX]
		ax1.plot(-ensemble_pred[1,:], ensemble_pred[0,:], '-m', marker='o', 
		        markersize=1, lw=0.5, alpha=0.5, label='Ensemble predictions' if INDEX == 0 else '')
	
	# Show predictions throughout simulation
	for idx, INDEX in enumerate(index_range_ensemble[:max_predictions]):
		if INDEX < n_ensemble_steps:
			ensemble_pred = ensemble_predictions[:,:,INDEX]
			ax1.plot(-ensemble_pred[1,:], ensemble_pred[0,:], '-m', marker='o', 
			        markersize=1, lw=0.5, alpha=0.5)
			ax1.scatter(-states_ensemble[1,INDEX], states_ensemble[0,INDEX], 
			          color='m', marker='o', alpha=0.6, s=15)
			ax1.text(-states_ensemble[1,INDEX]+0.05, states_ensemble[0,INDEX]+0.05, 
			        '{:.1f}'.format(time_ensemble[INDEX]), color='m', fontsize=9)

# Highlight where Baseline has large errors (failure regions)
baseline_failure_mask = lateral_errors_baseline > 0.15
if np.any(baseline_failure_mask):
	failure_indices = np.where(baseline_failure_mask)[0]
	# Only show a subset to avoid clutter
	failure_indices_subset = failure_indices[::max(1, len(failure_indices)//20)]
	ax1.scatter(-states_baseline[1, failure_indices_subset], states_baseline[0, failure_indices_subset],
	          c='red', s=40, marker='x', linewidths=2, alpha=0.7, 
	          label='Baseline large errors (>15cm)', zorder=4)

# Note: Weight estimation happens at EVERY step (not just at "adaptation points")
# The "adaptation points" shown are just where weights transition significantly
# This is for visualization only - the estimator runs continuously
if weights_ensemble.shape[0] >= 2:
	w_dry = weights_ensemble[0, :]
	w_ice = weights_ensemble[1, :]
	# Find significant weight transitions (for visualization)
	weight_change = np.diff(w_ice)
	# Find major transitions (>0.2 change) - these indicate switching between dry and ice
	major_transitions = np.where(np.abs(weight_change) > 0.2)[0] + 1
	if len(major_transitions) > 0:
		# Show all major transitions (not just first 3)
		# Filter to only show transitions that are significant and not too close together
		filtered_transitions = []
		last_idx = -10
		for idx in major_transitions:
			if idx - last_idx > 5:  # At least 5 steps apart
				filtered_transitions.append(idx)
				last_idx = idx
		
		if len(filtered_transitions) > 0:
			ax1.scatter(-states_ensemble[1, filtered_transitions], states_ensemble[0, filtered_transitions],
			          c='magenta', s=80, marker='*', alpha=0.9, edgecolors='k', linewidths=1.5,
			          label='Major weight transitions (dry↔ice)', zorder=4)
			# Add time labels for major transitions
			for idx in filtered_transitions[:5]:  # Label first 5
				if idx < len(time_ensemble):
					ax1.text(-states_ensemble[1,idx]+0.1, states_ensemble[0,idx]+0.1, 
					        f't={time_ensemble[idx]:.1f}s', color='magenta', fontsize=8, 
					        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax1.set_xlabel('$x$ [m] (plot: -y)', fontsize=14)
ax1.set_ylabel('$y$ [m] (plot: x)', fontsize=14)
ax1.set_title('Benchmark Comparison: Trajectories on ETHZ Track', fontsize=16, fontweight='bold')
ax1.legend(loc='best', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_RESULTS:
	plt.savefig(FIG1_PATH, dpi=300, bbox_inches='tight')
	print(f"   ✓ Saved to {FIG1_PATH}")
plt.close()


#####################################################################
# Figure 2: States & Errors

print("\n4. Generating Figure 2: States & Errors...")

fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Velocity vs Time
ax2a = axes[0]

# Oracle reference velocity (optimal with friction awareness)
if len(v_oracle) > 0:
	t_oracle_vec = np.linspace(0, t_oracle, len(v_oracle))
	v_oracle_interp = np.interp(time_ensemble, t_oracle_vec, v_oracle)
	ax2a.plot(time_ensemble, v_oracle_interp, 'b-', linewidth=2, alpha=0.8, 
	         label='Oracle Velocity (optimal with friction map)', zorder=3)

# Naive reference velocity (what MPC is trying to track - assumes dry everywhere)
if len(v_naive) > 0:
	t_naive_vec = np.linspace(0, t_naive, len(v_naive))
	v_naive_interp = np.interp(time_ensemble, t_naive_vec, v_naive)
	ax2a.plot(time_ensemble, v_naive_interp, 'g--', linewidth=2, alpha=0.7, 
	         label='Naive Reference Velocity (target, assumes μ=1.0)', zorder=2)

# Actual velocities
vx_baseline = states_baseline[3, :]
vx_ensemble = states_ensemble[3, :]

ax2a.plot(time_baseline, vx_baseline, 'r-', linewidth=2, label='Baseline MPC Velocity', zorder=1)
ax2a.plot(time_ensemble, vx_ensemble, 'm-', linewidth=2, label='Ensemble MPC Velocity', zorder=1)

# Highlight ice zones
ice_mask_baseline = friction_baseline < 0.5
ice_mask_ensemble = friction_ensemble < 0.5
if np.any(ice_mask_baseline):
	ax2a.fill_between(time_baseline, 0, 10, where=ice_mask_baseline, 
	                 alpha=0.15, color='cyan', label='Ice Zone')
if np.any(ice_mask_ensemble):
	ax2a.fill_between(time_ensemble, 0, 10, where=ice_mask_ensemble, 
	                 alpha=0.15, color='cyan')

ax2a.set_ylabel('Velocity [m/s]', fontsize=12)
ax2a.set_title('Velocity Comparison', fontsize=14, fontweight='bold')
ax2a.legend(loc='best', fontsize=10)
ax2a.grid(True, alpha=0.3)
ax2a.set_xlim([0, max(time_baseline[-1], time_ensemble[-1])])

# Plot 2: Lateral Error vs Time
ax2b = axes[1]

ax2b.plot(time_baseline, lateral_errors_baseline, 'r-', linewidth=2, label='Baseline MPC Error')
ax2b.plot(time_ensemble, lateral_errors_ensemble, 'm-', linewidth=2, label='Ensemble MPC Error')

# Failure threshold
ax2b.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Failure Threshold (0.2m)')

# Highlight ice zones
if np.any(ice_mask_baseline):
	ax2b.fill_between(time_baseline, 0, 0.5, where=ice_mask_baseline, 
	                 alpha=0.15, color='cyan')
if np.any(ice_mask_ensemble):
	ax2b.fill_between(time_ensemble, 0, 0.5, where=ice_mask_ensemble, 
	                 alpha=0.15, color='cyan')

ax2b.set_ylabel('Lateral Error [m]', fontsize=12)
ax2b.set_title('Lateral Tracking Error', fontsize=14, fontweight='bold')
ax2b.legend(loc='best', fontsize=10)
ax2b.grid(True, alpha=0.3)
ax2b.set_xlim([0, max(time_baseline[-1], time_ensemble[-1])])

# Plot 3: Friction over time
ax2c = axes[2]

ax2c.plot(time_baseline, friction_baseline, 'r-', linewidth=1.5, alpha=0.7, label='Baseline (actual)')
ax2c.plot(time_ensemble, friction_ensemble, 'm-', linewidth=1.5, alpha=0.7, label='Ensemble (actual)')
ax2c.axhline(y=0.3, color='cyan', linestyle='--', alpha=0.5, linewidth=1.5, label='Ice (μ=0.3)')
ax2c.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Asphalt (μ=1.0)')

ax2c.set_ylabel('Friction Coefficient μ', fontsize=12)
ax2c.set_xlabel('Time [s]', fontsize=12)
ax2c.set_title('Friction Map (Ice: y<-0.5, right in plot)', fontsize=14, fontweight='bold')
ax2c.legend(loc='best', fontsize=10)
ax2c.grid(True, alpha=0.3)
ax2c.set_xlim([0, max(time_baseline[-1], time_ensemble[-1])])

plt.tight_layout()
if SAVE_RESULTS:
	plt.savefig(FIG2_PATH, dpi=300, bbox_inches='tight')
	print(f"   ✓ Saved to {FIG2_PATH}")
plt.close()


#####################################################################
# Figure 3: Weight Adaptation

print("\n5. Generating Figure 3: Weight Adaptation...")

fig3, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Weight evolution
ax3a = axes[0]

if weights_ensemble.shape[0] >= 2:
	w_dry = weights_ensemble[0, :]
	w_ice = weights_ensemble[1, :]
	
	ax3a.plot(time_ensemble, w_dry, 'b-', linewidth=2.5, label='$w_{dry}$ (μ=1.0)', marker='o', markersize=3, alpha=0.8)
	ax3a.plot(time_ensemble, w_ice, 'c-', linewidth=2.5, label='$w_{ice}$ (μ=0.3)', marker='s', markersize=3, alpha=0.8)
	
	# Highlight ice zones
	if np.any(ice_mask_ensemble):
		ax3a.fill_between(time_ensemble, 0, 1, where=ice_mask_ensemble, 
		                 alpha=0.15, color='cyan', label='Ice Zone')
	
	ax3a.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
	ax3a.set_ylabel('Ensemble Weight', fontsize=12)
	ax3a.set_title('Weight Adaptation Over Time', fontsize=14, fontweight='bold')
	ax3a.legend(loc='best', fontsize=11)
	ax3a.grid(True, alpha=0.3)
	ax3a.set_ylim([-0.05, 1.05])
	ax3a.set_xlim([0, time_ensemble[-1]])

# Plot 2: Weight vs Friction (scatter)
ax3b = axes[1]

if weights_ensemble.shape[0] >= 2:
	# Color by friction value
	scatter1 = ax3b.scatter(friction_ensemble, w_dry, c=time_ensemble, cmap='viridis', 
	                       s=30, alpha=0.7, label='$w_{dry}$', edgecolors='k', linewidths=0.3)
	scatter2 = ax3b.scatter(friction_ensemble, w_ice, c=time_ensemble, cmap='plasma', 
	                       s=30, alpha=0.7, marker='s', label='$w_{ice}$', edgecolors='k', linewidths=0.3)
	
	# Expected relationship: high friction → high w_dry, low friction → high w_ice
	ax3b.axvline(x=0.3, color='cyan', linestyle='--', alpha=0.5, linewidth=1.5, label='Ice (μ=0.3)')
	ax3b.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Asphalt (μ=1.0)')
	
	ax3b.set_xlabel('Friction Coefficient μ', fontsize=12)
	ax3b.set_ylabel('Ensemble Weight', fontsize=12)
	ax3b.set_title('Weight vs Friction (colored by time)', fontsize=14, fontweight='bold')
	ax3b.legend(loc='best', fontsize=10)
	ax3b.grid(True, alpha=0.3)
	ax3b.set_ylim([-0.05, 1.05])
	
	# Add colorbar
	cbar = plt.colorbar(scatter1, ax=ax3b)
	cbar.set_label('Time [s]', fontsize=10)

plt.tight_layout()
if SAVE_RESULTS:
	plt.savefig(FIG3_PATH, dpi=300, bbox_inches='tight')
	print(f"   ✓ Saved to {FIG3_PATH}")
plt.close()


#####################################################################
# Summary

print("\n" + "=" * 60)
print("Plotting Complete!")
print("=" * 60)
print(f"\nGenerated figures:")
print(f"  1. {FIG1_PATH}")
print(f"  2. {FIG2_PATH}")
print(f"  3. {FIG3_PATH}")
