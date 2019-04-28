"""
created on Mon Apr. 23, 2019

@author: He Sun, Princeton University

wavefront estimators, including least-square batch process estimator (BPE), 
Kalman filter, extended Kalman filter.

"""
import numpy as np
import filters

class Batch_process:
	def __init__(self, params_values):
		self.params_values = params_values
		self.est_type = 'batch process estimator'

	def Estimate(self, If_p, u1p, u2p, exp_time):
		n_pair = u1p.shape[1]
		G1 = self.params_values['G1']
		G2 = self.params_values['G2']
		G = np.concatenate((G1, G2), axis=1)
		dI = np.empty((If_p.shape[0], If_p.shape[1], n_pair), dtype=float)
		for k in range(n_pair):
			dI[:, :, k] = If_p[:, :, 2*k] - If_p[:, :, 2*k+1]
		E_est = np.empty((If_p.shape[0], If_p.shape[1]), dtype=complex)
		P_est = np.empty((If_p.shape[0], 2, 2), dtype=float)
		for k in range(G1.shape[2]):
			G1m = G1[:, :, k]
			G2m = G2[:, :, k]
			dE = np.matmul(G1m, u1p) + np.matmul(G2m, u2p)
			contrast_p = np.mean(If_p[:, k, :], 0)
			d_contrast_p_half = np.mean(np.abs(dE)**2, 0)
			d_contrast_p = np.empty(contrast_p.shape)
			d_contrast_p[0::2] = d_contrast_p_half
			d_contrast_p[1::2] = d_contrast_p_half
			cov_p = self.params_values['R0'] / exp_time**2 + \
					self.params_values['R1'] / exp_time * contrast_p + \
					4*(self.params_values['Q0'] + self.params_values['Q1']*d_contrast_p) * contrast_p
			R = np.diag(cov_p[0::2] + cov_p[1::2])
			for i in range(G.shape[0]):
				H = np.empty((dE.shape[1], 2), dtype=float)
				for l in range(dE.shape[1]):
					H[l, :] = 4 * np.array([dE[i, l].real, dE[i, l].imag])
				y = dI[i, k , :]
				x_hat, P_est_now = filters.lse(y, H, R)
				E_est[i, k] = x_hat[0] + 1j * x_hat[1]
				P_est[i, :, :] = P_est_now
		return E_est, P_est


class Kalman_filter:
	def __init__(self, params_values):
		self.params_values = params_values
		self.est_type = 'Kalman filter'

	def Estimate(self, If_p, u1p, u2p, Enp_old, P_old, u1c, u2c, exp_time):
		n_pair = u1p.shape[1]
		G1 = self.params_values['G1']
		G2 = self.params_values['G2']
		G = np.concatenate((G1, G2), axis=1)
		uc = np.concatenate([u1c, u2c])
		dI = np.empty((If_p.shape[0], If_p.shape[1], n_pair), dtype=float)
		for k in range(n_pair):
			dI[:, :, k] = If_p[:, :, 2*k] - If_p[:, :, 2*k+1]
		E_est = np.empty((If_p.shape[0], If_p.shape[1]), dtype=complex)
		P_est = np.empty((If_p.shape[0], 2, 2), dtype=float)
		for k in range(G1.shape[2]):
			G1m = G1[:, :, k]
			G2m = G2[:, :, k]
			Gm = G[:, :, k]
			dE = np.matmul(G1m, u1p) + np.matmul(G2m, u2p)
			Q = (self.params_values['Q1'] * np.mean(np.abs(dE)**2) + \
				self.params_values['Q0']) * np.eye(2)
			contrast_p = np.mean(If_p[:, k, :], 0)
			d_contrast_p_half = np.mean(np.abs(dE)**2, 0)
			d_contrast_p = np.empty(contrast_p.shape)
			d_contrast_p[0::2] = d_contrast_p_half
			d_contrast_p[1::2] = d_contrast_p_half
			cov_p = self.params_values['R0'] / exp_time**2 + \
					self.params_values['R1'] / exp_time * contrast_p + \
					4*(self.params_values['Q0'] + self.params_values['Q1']*d_contrast_p) * contrast_p
			R = np.diag(cov_p[0::2] + cov_p[1::2])
			for i in range(G.shape[0]):
				H = np.empty((dE.shape[1], 2), dtype=float)
				for l in range(dE.shape[1]):
					H[l, :] = 4 * np.array([dE[i, l].real, dE[i, l].imag])
				G_now = np.concatenate([Gm[i, :].real.reshape((1, -1)), 
										Gm[i, :].real.reshape((1, -1))], 0)
				y = dI[i, k , :]
				x_old = np.array([Enp_old[i].real, Enp_old[i].imag])
				P_est_old = P_old[i, :, :]
				x_hat, P_est_now, _, _ = filters.Kalman_filter(y, uc, x_old, P_est_old, 
														np.eye(2), G_now, H, Q, R)
				E_est[i, k] = x_hat[0] + 1j * x_hat[1]
				P_est[i, :, :] = P_est_now
		return E_est, P_est
