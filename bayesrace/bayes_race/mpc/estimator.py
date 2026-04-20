"""	Ensemble Weight Estimator using Linear Regression on State Derivatives.

This module implements a weight estimator for ensemble dynamics models.
It uses linear regression on observed vs predicted state derivatives to
estimate the weights w_i such that:
    dxdt_obs ≈ sum_i w_i * dxdt_pred_i

Subject to constraints: sum(w_i) = 1, w_i >= 0
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
from scipy.optimize import lsq_linear
from scipy.optimize import minimize


class EnsembleWeightEstimator:
	"""Estimator for ensemble model weights using linear regression on derivatives."""

	def __init__(self, models, buffer_size=10, Ts=0.02):
		"""Initialize estimator.
		
		Args:
			models: List of Dynamic model instances (ensemble)
			buffer_size: Number of recent steps to use for estimation (M)
			Ts: Sampling time
		"""
		self.models = models
		self.n_models = len(models)
		self.buffer_size = buffer_size
		self.Ts = Ts
		
		# Buffers for states and inputs
		self.state_buffer = []
		self.input_buffer = []
		
		# Current weight estimate (initialize uniform)
		self.weights = np.ones(self.n_models) / self.n_models

	def update_buffer(self, x, u):
		"""Add new state and input to buffer.
		
		Args:
			x: State vector (n_states,) or (n_states, 1)
			u: Input vector (n_inputs,) or (n_inputs, 1)
		"""
		x = np.asarray(x).flatten()
		u = np.asarray(u).flatten()
		
		self.state_buffer.append(x.copy())
		self.input_buffer.append(u.copy())
		
		# Maintain buffer size
		if len(self.state_buffer) > self.buffer_size:
			self.state_buffer.pop(0)
			self.input_buffer.pop(0)

	def _compute_observed_derivatives(self):
		"""Compute observed derivatives from state buffer.
		
		Returns:
			dxdt_obs: Observed derivatives (buffer_size-1, n_states)
		"""
		if len(self.state_buffer) < 2:
			return None
		
		n_states = len(self.state_buffer[0])
		n_steps = len(self.state_buffer) - 1
		dxdt_obs = np.zeros((n_steps, n_states))
		
		for k in range(n_steps):
			dxdt_obs[k, :] = (self.state_buffer[k+1] - self.state_buffer[k]) / self.Ts
		
		return dxdt_obs

	def _compute_predicted_derivatives(self, model, x, u):
		"""Compute predicted derivatives for a single model.
		
		Args:
			model: Dynamic model instance
			x: State vector (n_states,)
			u: Input vector (n_inputs,)
		
		Returns:
			dxdt: Predicted derivative (n_states,)
		"""
		# Use the model's _diffequation method to compute derivatives
		# This avoids full simulation and gives us the derivative directly
		dxdt = model._diffequation(None, x, u)
		return dxdt

	def estimate_weights(self):
		"""Estimate ensemble weights using constrained least squares.
		
		Returns:
			weights: Estimated weights (n_models,)
		"""
		if len(self.state_buffer) < 2:
			# Not enough data, return current estimate
			return self.weights.copy()
		
		# Compute observed derivatives
		dxdt_obs = self._compute_observed_derivatives()
		if dxdt_obs is None:
			return self.weights.copy()
		
		n_steps, n_states = dxdt_obs.shape
		
		# Compute predicted derivatives for each model
		dxdt_pred_list = []
		for model in self.models:
			dxdt_pred_i = np.zeros((n_steps, n_states))
			for k in range(n_steps):
				dxdt_pred_i[k, :] = self._compute_predicted_derivatives(
					model, self.state_buffer[k], self.input_buffer[k]
				)
			dxdt_pred_list.append(dxdt_pred_i)
		
		# Stack predictions: A[i, :] = flattened predictions from model i
		# Shape: (n_steps * n_states, n_models)
		A = np.zeros((n_steps * n_states, self.n_models))
		for i in range(self.n_models):
			A[:, i] = dxdt_pred_list[i].flatten()
		
		# Observed derivatives: b = flattened observed derivatives
		# Shape: (n_steps * n_states,)
		b = dxdt_obs.flatten()
		
		# Solve constrained least squares:
		# min ||A @ w - b||^2
		# s.t. sum(w) = 1, w >= 0
		
		# Use scipy.optimize.minimize with constraints
		def objective(w):
			return np.sum((A @ w - b)**2)
		
		# Constraints: sum(w) = 1
		constraints = {
			'type': 'eq',
			'fun': lambda w: np.sum(w) - 1.0
		}
		
		# Bounds: w >= 0
		bounds = [(0.0, 1.0) for _ in range(self.n_models)]
		
		# Initial guess: current weights or uniform
		w0 = self.weights.copy()
		
		# Solve
		result = minimize(
			objective,
			w0,
			method='SLSQP',
			bounds=bounds,
			constraints=constraints,
			options={'maxiter': 100, 'ftol': 1e-6}
		)
		
		if result.success:
			self.weights = result.x
		else:
			# If optimization fails, use uniform weights as fallback
			print(f"Warning: Weight estimation failed, using uniform weights. Error: {result.message}")
			self.weights = np.ones(self.n_models) / self.n_models
		
		return self.weights.copy()

	def get_weights(self):
		"""Get current weight estimate.
		
		Returns:
			weights: Current weights (n_models,)
		"""
		return self.weights.copy()

	def reset(self):
		"""Reset buffers and weights."""
		self.state_buffer = []
		self.input_buffer = []
		self.weights = np.ones(self.n_models) / self.n_models
