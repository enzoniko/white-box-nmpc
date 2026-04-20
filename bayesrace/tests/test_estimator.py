"""Unit test for Ensemble Weight Estimator.

This test validates that the estimator can recover correct weights when
given data generated from a known model.
"""

import sys
import os
import numpy as np

# Add bayesrace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bayes_race.models import Dynamic
from bayes_race.params import ORCA
from bayes_race.mpc.estimator import EnsembleWeightEstimator


def test_estimator_recovers_ice_model():
	"""Test that estimator recovers weights close to [0.0, 1.0] for pure ice model simulation."""
	
	print("=" * 60)
	print("Test: Estimator Recovers Ice Model Weights")
	print("=" * 60)
	
	# Setup: Create two models (Dry and Ice)
	print("\n1. Creating ensemble models...")
	params = ORCA(control='pwm')
	
	# Dry model: μ = 1.0 (default)
	model_dry = Dynamic(**params)
	print(f"   ✓ Dry model: Df={params['Df']:.4f}, Dr={params['Dr']:.4f}")
	
	# Ice model: μ = 0.3 (scale Df and Dr)
	params_ice = params.copy()
	params_ice['Df'] = params['Df'] * 0.3
	params_ice['Dr'] = params['Dr'] * 0.3
	model_ice = Dynamic(**params_ice)
	print(f"   ✓ Ice model: Df={params_ice['Df']:.4f}, Dr={params_ice['Dr']:.4f}")
	
	# Simulate using ONLY ice model (ground truth)
	print("\n2. Simulating 10 steps using ONLY ice model...")
	Ts = 0.02
	n_steps = 10
	
	# Initial state: [x, y, psi, vx, vy, omega]
	x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
	
	# Generate random but reasonable inputs
	np.random.seed(42)  # For reproducibility
	u_seq = np.zeros((2, n_steps))
	u_seq[0, :] = np.random.uniform(0.3, 0.7, n_steps)  # PWM: 0.3 to 0.7
	u_seq[1, :] = np.random.uniform(-0.1, 0.1, n_steps)  # Steering: small angles
	
	# Time vector
	time = np.linspace(0, n_steps * Ts, n_steps + 1)
	
	# Simulate using ice model
	x_seq, dxdt_seq = model_ice.sim_continuous(x0, u_seq, time)
	print(f"   ✓ Simulated {n_steps} steps")
	print(f"   Final state: x={x_seq[0, -1]:.3f}, y={x_seq[1, -1]:.3f}, vx={x_seq[3, -1]:.3f}")
	
	# Feed data to estimator
	print("\n3. Feeding data to estimator...")
	estimator = EnsembleWeightEstimator(
		models=[model_dry, model_ice],
		buffer_size=10,
		Ts=Ts
	)
	
	# Update buffer with all steps
	for k in range(n_steps + 1):
		estimator.update_buffer(x_seq[:, k], u_seq[:, k] if k < n_steps else u_seq[:, -1])
	
	print(f"   ✓ Buffer updated with {len(estimator.state_buffer)} states")
	
	# Estimate weights
	print("\n4. Estimating weights...")
	weights = estimator.estimate_weights()
	print(f"   Estimated weights: w_dry={weights[0]:.4f}, w_ice={weights[1]:.4f}")
	
	# Expected weights: [0.0, 1.0] (all weight on ice model)
	expected_weights = np.array([0.0, 1.0])
	
	# Assertion
	print("\n5. Validating results...")
	tolerance = 0.15  # Allow some tolerance due to discretization and numerical errors
	error = np.abs(weights - expected_weights)
	max_error = np.max(error)
	
	print(f"   Expected weights: {expected_weights}")
	print(f"   Estimated weights: {weights}")
	print(f"   Max error: {max_error:.4f}")
	print(f"   Tolerance: {tolerance}")
	
	if max_error < tolerance:
		print(f"\n✓ TEST PASSED: Estimator recovered weights within tolerance!")
		return True
	else:
		print(f"\n✗ TEST FAILED: Estimator did not recover weights within tolerance")
		print(f"   Error: {error}")
		return False


def test_estimator_recovers_dry_model():
	"""Test that estimator recovers weights close to [1.0, 0.0] for pure dry model simulation."""
	
	print("\n" + "=" * 60)
	print("Test: Estimator Recovers Dry Model Weights")
	print("=" * 60)
	
	# Setup: Create two models (Dry and Ice)
	print("\n1. Creating ensemble models...")
	params = ORCA(control='pwm')
	
	# Dry model: μ = 1.0 (default)
	model_dry = Dynamic(**params)
	
	# Ice model: μ = 0.3
	params_ice = params.copy()
	params_ice['Df'] = params['Df'] * 0.3
	params_ice['Dr'] = params['Dr'] * 0.3
	model_ice = Dynamic(**params_ice)
	
	# Simulate using ONLY dry model (ground truth)
	print("\n2. Simulating 20 steps using ONLY dry model with aggressive inputs...")
	Ts = 0.02
	n_steps = 20  # More steps
	
	x0 = np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0])  # Higher initial velocity
	np.random.seed(123)  # Different seed for variety
	u_seq = np.zeros((2, n_steps))
	# Much more aggressive inputs to distinguish dry from ice
	u_seq[0, :] = np.random.uniform(0.5, 0.9, n_steps)  # Higher PWM
	u_seq[1, :] = np.random.uniform(-0.25, 0.25, n_steps)  # Larger steering angles
	
	time = np.linspace(0, n_steps * Ts, n_steps + 1)
	x_seq, dxdt_seq = model_dry.sim_continuous(x0, u_seq, time)
	print(f"   ✓ Simulated {n_steps} steps")
	print(f"   Final state: x={x_seq[0, -1]:.3f}, y={x_seq[1, -1]:.3f}, vx={x_seq[3, -1]:.3f}")
	
	# Feed to estimator
	print("\n3. Feeding data to estimator...")
	estimator = EnsembleWeightEstimator(
		models=[model_dry, model_ice],
		buffer_size=20,  # Match number of steps
		Ts=Ts
	)
	
	for k in range(n_steps + 1):
		estimator.update_buffer(x_seq[:, k], u_seq[:, k] if k < n_steps else u_seq[:, -1])
	
	# Estimate weights
	print("\n4. Estimating weights...")
	weights = estimator.estimate_weights()
	print(f"   Estimated weights: w_dry={weights[0]:.4f}, w_ice={weights[1]:.4f}")
	
	# Expected weights: [1.0, 0.0]
	expected_weights = np.array([1.0, 0.0])
	
	# Assertion
	print("\n5. Validating results...")
	# Note: Dry model test is more challenging because with small inputs,
	# both models can predict similar derivatives. We use a more lenient tolerance.
	tolerance = 0.25  # More tolerance for dry model (harder to distinguish)
	error = np.abs(weights - expected_weights)
	max_error = np.max(error)
	
	print(f"   Expected weights: {expected_weights}")
	print(f"   Estimated weights: {weights}")
	print(f"   Max error: {max_error:.4f}")
	
	if max_error < tolerance:
		print(f"\n✓ TEST PASSED: Estimator recovered weights within tolerance!")
		return True
	else:
		print(f"\n✗ TEST FAILED: Estimator did not recover weights within tolerance")
		return False


if __name__ == '__main__':
	print("\n" + "=" * 60)
	print("Phase 3: Ensemble Weight Estimator Unit Tests")
	print("=" * 60)
	
	# Run tests
	test1_passed = test_estimator_recovers_ice_model()
	test2_passed = test_estimator_recovers_dry_model()
	
	# Summary
	print("\n" + "=" * 60)
	print("Test Summary")
	print("=" * 60)
	print(f"Test 1 (Ice Model): {'PASSED' if test1_passed else 'FAILED'}")
	print(f"Test 2 (Dry Model): {'PASSED' if test2_passed else 'FAILED'}")
	
	if test1_passed and test2_passed:
		print("\n✓ ALL TESTS PASSED!")
		sys.exit(0)
	else:
		print("\n✗ SOME TESTS FAILED")
		sys.exit(1)
