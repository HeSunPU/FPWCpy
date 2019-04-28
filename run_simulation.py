import scipy.io as sio
import scipy as sp
import numpy as np
import SPLC_model as splc
import VORTEX_model as vortex
import systemID as sysid
import estimation as est
import sensing
import detector
import helper_function as hp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.ion()


def EFC(x, G, weight, alpha):
	M = np.zeros((G.shape[1], G.shape[1]))
	Gx = np.zeros((G.shape[1], 1))
	for k in range(len(weight)):
		Gx += weight[k] * np.real(np.matmul(np.conj(G[:, :, k].T), x[:, k].reshape((-1, 1))))
		M += weight[k] * np.real(np.matmul(np.conj(G[:, :, k].T), G[:, :, k]))
	command = -np.real(np.matmul(np.linalg.inv(M + alpha*np.eye(G.shape[1])), Gx.reshape((-1, 1))))
	return command.reshape(G.shape[1])


if __name__ == "__main__":
	# define the optical model
	wavelength = 635e-9 * np.ones(1)
	DM1gain = 5.06e-9 * np.ones((34, 34))
	DM2gain = 6.27e-9 * np.ones((34, 34))
	coronagraph_type = 'vortex'#'splc'#

	if coronagraph_type.lower() == 'splc':
		model = splc.Optical_Model(wavelength, DM1gain, DM2gain, wfe=True)
		model_perfect = splc.Optical_Model(wavelength, DM1gain, DM2gain, wfe=False)
		# define the dark hole region
		dh_ind1, dh_ind2 = hp.Compute_DH(model, dh_shape='wedge', range_r=[2.5, 9], range_angle=30)
	elif coronagraph_type.lower() == 'vortex':
		model = vortex.Optical_Model(wavelength, DM1gain, DM2gain, wfe=True)
		model_perfect = vortex.Optical_Model(wavelength, DM1gain, DM2gain, wfe=False)
		dh_ind1, dh_ind2 = hp.Compute_DH(model, dh_shape='circ', range_r=[3, 9], range_angle=30)



	# compute or load the Jacobian matrices
	# G1, G2 = hp.Compute_Jacobian(model_perfect, dh_ind1, dh_ind2, print_flag=True)

	G1 = np.load('vortex_compact_Jacobian1.npy')
	G2 = np.load('vortex_compact_Jacobian2.npy')

	# G1 = np.load('splc_Jacobian1.npy')
	# G2 = np.load('splc_Jacobian2.npy')

	G = np.concatenate((G1, G2), axis=1)


	# define the control parameters
	Nitr = 20 # number of control iterations
	n_trials = 1
	n_act = G1.shape[1] # number of active actuators on the DM
	n_pix = G1.shape[0] # number of pixels in the dark hole
	weight = np.ones(len(wavelength))
	alpha = 3e-7
	img_number = 8
	exp_time = 10

	# define the wavefront estimator
	params_values = {}
	params_values['G1'] = G1
	params_values['G2'] = G2
	params_values['Q0'] = 1e-12
	params_values['Q1'] = 0.15 # 0.05 # 0.5 # 1e-8 for u^2, 6e-8 for u^3, 0.5 for dE^2
	params_values['R0'] = 3.6e-17#/exp_time**2 #1e-14
	params_values['R1'] = 5e-10#/exp_time #1e-9
	params_values['R2'] = 4 * params_values['Q1']#4 * 0.045 #5e-3 # 5e-3 # 1e-2
	BPE_estimator = est.Batch_process(params_values)
	KF_estimator = est.Kalman_filter(params_values)

	# define the sensing policy
	sensor0 = sensing.Empirical_probe(model, params_values, img_number, 
									pair_wise=True, probe_area=[1, 17, -17, 17], method='alter')
	sensor = sensing.Optimal_probe(params_values, img_number, pair_wise=True)

	# define the camera noise model
	camera = detector.CCD(flux=2e9, readout_std=12, readout=True, photon=True, exp_time=exp_time)

	# define the system identifier for adaptive control
	# n_batch = 5
	# vl = sysid.linear_vl(params_values, img_number//2)

	# decide whether to save the wfsc data
	save_data_flag = True
	if save_data_flag:
		data_train = {}
		data_train['u1'] = np.ones((n_act, Nitr))
		data_train['u2'] = np.ones((n_act, Nitr))
		data_train['u1p'] = np.ones((n_act, img_number, Nitr))
		data_train['u2p'] = np.ones((n_act, img_number, Nitr))
		data_train['I'] = np.ones((n_pix, img_number+1, Nitr))
		data_train['time'] = np.zeros(Nitr)

	# Ef_est_set = []
	# Ef_vector_set = []
	# P_est_set = []
	# contrast_set = []

	# for trial in range(n_trials):
	# start wavefront control
	u1 = np.zeros((34, 34))
	u2 = np.zeros((34, 34))
	u10 = u1
	u20 = u2
	contrast = np.zeros((Nitr, ))
	E_true = []
	for k in range(Nitr):
		# collect the unprobed image
		Ef = model.Propagate(u1, u2)
		# Ef = sp.ndimage.shift(Ef.real, shift=[0.5, 0.5, 0]) + 1j * sp.ndimage.shift(Ef.imag, shift=[0.5, 0.5, 0])
		Ef_vector = Ef[dh_ind1, dh_ind2, :]
		If_vector = np.abs(Ef_vector)**2
		If = camera.Add_noise(If_vector)
		contrast[k] = np.mean(If_vector)
		print('The contrast at step #{} is {}'.format(k, contrast[k]))
		E_true.append(Ef_vector)
		
		# collect probe images
		# camera.set_exposure(np.max([1e-9/contrast[k], 1e-3]))
		# exp_time = camera.exp_time
		print('The exposure time at step #{} is {}'.format(k, exp_time))

		R_coef = [params_values['R0']/exp_time**2, params_values['R1']/exp_time, params_values['R2']]
		u_p = sensor0.Probe_command(contrast[k], k, index=1, R_coef=R_coef)
		u_p_values = u_p[model.DMind1, model.DMind2, :].T

		contrast_p = np.sqrt(params_values['R0']/exp_time**2/params_values['R2'] + \
					(params_values['R1']/exp_time + 4*params_values['Q0'])*contrast[k]/params_values['R2'])
		print('designed probe contrast is {}.'.format(contrast_p))

		u_p = np.zeros((model.Nact, model.Nact, img_number//2))
		# u_p_values = 3e2 * np.sqrt(contrast_p) * np.random.rand(img_number//2, n_act)
		u_p_values = sensor.Probe_command(u_p_values, exp_time, contrast[k], rate=5e-4, beta=3e-1, gamma=1., Nitr=1000, print_flag=True)
		u_p[model.DMind1, model.DMind2, :] = u_p_values.T

		
		# , R_coef=[params_values['R0'], params_values['R1'], params_values['R2']]
		If_p = np.empty((len(dh_ind1), len(wavelength), img_number), dtype=float)
		Ef_p_set = np.empty((len(dh_ind1), len(wavelength), img_number), dtype=complex)
		for i in range(u_p.shape[2]):
			# images with positive probes
			Ef_p = model.Propagate(u1+u_p[:, :, i], u2)
			# Ef_p = sp.ndimage.shift(Ef_p.real, shift=[0.5, 0.5, 0]) + 1j * sp.ndimage.shift(Ef_p.imag, shift=[0.5, 0.5, 0])
			Ef_p_vector = Ef_p[dh_ind1, dh_ind2, :]
			If_p_vector = np.abs(Ef_p_vector)**2
			# If_p[:, :, 2*i] = If_p_vector
			If_p[:, :, 2*i] = camera.Add_noise(If_p_vector)
			Ef_p_set[:, :, 2*i] = Ef_p_vector
			print('The contrast of the No.{} postive image is {}'.format(i, np.mean(If_p_vector)))
			# images with negative probes
			Ef_p = model.Propagate(u1-u_p[:, :, i], u2)
			# Ef_p = sp.ndimage.shift(Ef_p.real, shift=[0.5, 0.5, 0]) + 1j * sp.ndimage.shift(Ef_p.imag, shift=[0.5, 0.5, 0])
			Ef_p_vector = Ef_p[dh_ind1, dh_ind2, :]
			If_p_vector = np.abs(Ef_p_vector)**2
			# If_p[:, :, 2*i+1] = If_p_vector
			If_p[:, :, 2*i+1] = camera.Add_noise(If_p_vector)
			Ef_p_set[:, :, 2*i+1] = Ef_p_vector
			print('The contrast of the No.{} negative image is {}'.format(i, np.mean(If_p_vector)))
			# print('The contrast of the No.{} difference image is {}'.format(i, np.mean(np.abs(If_p[:, :, 2*i] - If_p[:, :, 2*i+1]))))

		# Ef_est = Ef_vector
		u_p_vector = u_p[model.DMind1, model.DMind2, :]
		if k >= 0:
			Ef_est, P_est = BPE_estimator.Estimate(If_p, u_p_vector, np.zeros(u_p_vector.shape), exp_time)
		else:
			Ef_est, P_est = KF_estimator.Estimate(If_p, u_p_vector, np.zeros(u_p_vector.shape), Ef_est, P_est, 
												command[0:n_act], command[n_act::], exp_time)
		# compute control command
		command = EFC(Ef_est, G, weight, alpha)
		u1[model.DMind1, model.DMind2] += command[:int(len(command)/2):]
		u2[model.DMind1, model.DMind2] += command[int(len(command)/2)::]

		data_train['u1'][:, k] = command[0:n_act]
		data_train['u2'][:, k]= command[n_act::]
		for k_image in range(img_number):
			data_train['u1p'][:, k_image, k] = ((-1)**(k_image%2)) * u_p[model.DMind1, model.DMind2, k_image//2]
			data_train['u2p'][:, k_image, k] = np.zeros((n_act, ))
		data_train['I'][:, 0, k] = np.squeeze(If)
		data_train['I'][:, 1::, k] = np.squeeze(If_p)
		data_train['time'][k] = camera.exp_time

		# Ef_est_set.append(Ef_est)
		# Ef_vector_set.append(Ef_vector)
		# P_est_set.append(P_est)

	# 	if (k+1) % n_batch == 0:
	# 		data_train_now = {}
	# 		data_train_now['u1'] = data_train['u1'][:, k+1-n_batch:k+1]
	# 		data_train_now['u2'] = data_train['u2'][:, k+1-n_batch:k+1]
	# 		data_train_now['u1p'] = data_train['u1p'][:, :, k+1-n_batch:k+1]
	# 		data_train_now['u2p'] = data_train['u2p'][:, :, k+1-n_batch:k+1]
	# 		data_train_now['I'] = data_train['I'][:, :, k+1-n_batch:k+1]
	# 		mse_list = vl.train_params(data_train_now, lr=1e-8, lr2=1e-3, epoch=30, 
	# 						params_trainable='all', print_flag=True)
	# 		G1 = params_values['G1']
	# 		G2 = params_values['G2']
	# 		G = np.concatenate((G1, G2), axis=1)

	

	# contrast_set.append(contrast)

			