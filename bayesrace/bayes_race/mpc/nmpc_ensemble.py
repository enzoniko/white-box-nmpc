"""	Setup NLP in CasADi with Ensemble Dynamics (Weighted Sum of Multiple Models).

This module implements an ensemble-based MPC formulation where the dynamics
are a weighted sum of multiple physics models:
    x_{next} = sum_{i=1}^N w_i * f_i(x, u; theta_i)

The weights w are passed as parameters to the solver, allowing online adaptation.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import casadi as cs

from bayes_race.mpc.constraints import Boundary


class setupNLPEnsemble:

	def __init__(self, horizon, Ts, Q, P, R, params, models, track, track_cons=False):
		"""Initialize ensemble-based MPC NLP.
		
		Args:
			horizon: Prediction horizon
			Ts: Sampling time
			Q: State cost matrix (2x2, for x,y)
			P: Terminal state cost matrix (2x2, for x,y)
			R: Input cost matrix (2x2)
			params: Vehicle parameters dict (for constraints)
			models: List of Dynamic model instances (ensemble)
			track: Track object
			track_cons: Whether to use track constraints
		"""
		self.horizon = horizon
		self.params = params
		self.models = models  # List of models
		self.n_models = len(models)
		self.track = track
		self.track_cons = track_cons

		# Use first model to get dimensions (all models should have same dimensions)
		model = models[0]
		n_states = model.n_states
		n_inputs = model.n_inputs
		xref_size = 2

		# Verify all models have same dimensions
		for i, m in enumerate(models):
			if m.n_states != n_states or m.n_inputs != n_inputs:
				raise ValueError(f"Model {i} has incompatible dimensions: "
				               f"states={m.n_states}, inputs={m.n_inputs}")

		# CasADi variables
		x0 = cs.SX.sym('x0', n_states, 1)
		xref = cs.SX.sym('xref', xref_size, horizon+1)
		uprev = cs.SX.sym('uprev', 2, 1)
		x = cs.SX.sym('x', n_states, horizon+1)
		u = cs.SX.sym('u', n_inputs, horizon)
		dxdtc = cs.SX.sym('dxdt', n_states, 1)
		
		# Weights parameter: w[i] is weight for model i
		w = cs.SX.sym('w', self.n_models, 1)

		if track_cons:
			eps = cs.SX.sym('eps', 2, horizon)
			Aineq = cs.SX.sym('Aineq', 2*horizon, 2)
			bineq = cs.SX.sym('bineq', 2*horizon, 1)		

		# Sum problem objectives and concatenate constraints
		cost_tracking = 0
		cost_actuation = 0
		cost_violation = 0

		cost_tracking += (x[:xref_size,-1]-xref[:xref_size,-1]).T @ P @ (x[:xref_size,-1]-xref[:xref_size,-1])
		constraints = x[:,0] - x0

		# Dynamics constraints: weighted sum of ensemble models
		for idh in range(horizon):
			# Compute weighted sum of dynamics from all models
			dxdt_ensemble = cs.SX.zeros(n_states, 1)
			for i, model in enumerate(self.models):
				dxdt_i = model.casadi(x[:,idh], u[:,idh], dxdtc)
				dxdt_ensemble += w[i] * dxdt_i
			
			constraints = cs.vertcat(constraints, x[:,idh+1] - x[:,idh] - Ts*dxdt_ensemble)

		for idh in range(horizon):

			# Delta between subsequent time steps
			if idh==0:
				deltaU  = u[:,idh]-uprev
			else:
				deltaU = u[:,idh]-u[:,idh-1]

			cost_tracking += (x[:xref_size,idh+1]-xref[:xref_size,idh+1]).T @ Q @ (x[:xref_size,idh+1]-xref[:xref_size,idh+1])
			cost_actuation += deltaU.T @ R @ deltaU

			if track_cons:
				cost_violation += 1e6 * (eps[:,idh].T @ eps[:,idh])

			constraints = cs.vertcat(constraints, u[:,idh] - params['max_inputs'])
			constraints = cs.vertcat(constraints, -u[:,idh] + params['min_inputs'])
			constraints = cs.vertcat(constraints, deltaU[1] - params['max_rates'][1]*Ts)
			constraints = cs.vertcat(constraints, -deltaU[1] + params['min_rates'][1]*Ts)

			# Track constraints
			if track_cons:
				constraints = cs.vertcat(constraints, Aineq[2*idh:2*idh+2,:] @ x[:2,idh+1] - bineq[2*idh:2*idh+2,:] - eps[:,idh])
		
		cost = cost_tracking + cost_actuation + cost_violation

		xvars = cs.vertcat(
			cs.reshape(x,-1,1),
			cs.reshape(u,-1,1),
			)
		if track_cons:
			xvars = cs.vertcat(
				xvars,
				cs.reshape(eps,-1,1),
				)

		pvars = cs.vertcat(
			cs.reshape(x0,-1,1), 
			cs.reshape(xref,-1,1), 
			cs.reshape(uprev,-1,1),
			cs.reshape(w,-1,1),  # Add weights to parameters
			)
		if track_cons:
			pvars = cs.vertcat(
				pvars,
				cs.reshape(Aineq,-1,1),
				cs.reshape(bineq,-1,1),
				)

		nlp = {
			'x': xvars,
			'p': pvars,
			'f': cost, 
			'g': constraints,
			}
		ipoptoptions = {
			'print_level': 0,
			'print_timing_statistics': 'no',
			'max_iter': 100,
			}
		options = {
			'expand': True,
			'print_time': False,
			'ipopt': ipoptoptions,
		}
		name = 'nmpc_ensemble'
		self.problem = cs.nlpsol(name, 'ipopt', nlp, options)

	def solve(self, x0, xref, uprev, weights):
		"""Solve MPC optimization with ensemble weights.
		
		Args:
			x0: Initial state (n_states,)
			xref: Reference trajectory (2, horizon+1)
			uprev: Previous input (2,)
			weights: Ensemble weights (n_models,) - should sum to 1, all >= 0
		
		Returns:
			umpc: Optimal input sequence (n_inputs, horizon)
			fval: Optimal cost value
			xmpc: Optimal state sequence (n_states, horizon+1)
		"""
		n_states = self.models[0].n_states
		n_inputs = self.models[0].n_inputs
		horizon = self.horizon
		track_cons = self.track_cons

		# Normalize weights to ensure they sum to 1
		weights = np.asarray(weights).flatten()
		weights = weights / (weights.sum() + 1e-10)  # Avoid division by zero
		weights = weights.reshape(-1, 1)

		# Track constraints
		if track_cons:
			Aineq = np.zeros([2*horizon,2])
			bineq = np.zeros([2*horizon,1])
			for idh in range(horizon):
				Ain, bin = Boundary(xref[:2,idh+1], self.track)
				Aineq[2*idh:2*idh+2,:] = Ain
				bineq[2*idh:2*idh+2] = bin
		else:
			Aineq = np.zeros([0,2])
			bineq = np.zeros([0,1])

		arg = {}
		arg['p'] = np.concatenate([
			x0.reshape(-1,1), 
			xref.T.reshape(-1,1), 
			uprev.reshape(-1,1),
			weights,  # Add weights to parameter vector
			])
		if track_cons:
			arg['p'] = np.concatenate([
				arg['p'],
				Aineq.T.reshape(-1,1),
				bineq.T.reshape(-1,1),			
				])
		
		arg['lbx'] = -np.inf*np.ones(n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons)
		arg['ubx'] = np.inf*np.ones(n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons)
		arg['lbg'] = np.concatenate([np.zeros(n_states*(horizon+1)), -np.inf*np.ones(horizon*(6+2*track_cons))])
		arg['ubg'] = np.concatenate([np.zeros(n_states*(horizon+1)), np.zeros(horizon*(6+2*track_cons))])
		
		res = self.problem(**arg)
		fval = res['f'].full()[0][0]
		xmpc = res['x'][:n_states*(horizon+1)].full().reshape(horizon+1,n_states).T
		umpc = res['x'][n_states*(horizon+1):n_states*(horizon+1)+n_inputs*horizon].full().reshape(horizon,n_inputs).T
		return umpc, fval, xmpc
