import scipy.io as sio
import numpy as np
import EMsystemID as em
import time
import os


if __name__ == "__main__":
	# define the EM algorithm system identifier
	n_pix = 188
	n_act = 952
	n_pair = 2
	n_waves = 5
	n_EMitr = 3#10
	folder = 'C:/Lab/FPWCmatlab/dataLibrary/20191015/'
	model_type = 'reduced' # 'normal' or 'reduced'
	model_dim = 200

	modelIFS = sio.loadmat('C:/Lab/FPWCmatlab/modelIFS.mat')

	G1_broadband = modelIFS['modelIFS']['G1'][0, 0]
	G2_broadband = modelIFS['modelIFS']['G2'][0, 0]

	G1_list = []
	G2_list = []

	for k in range(n_waves):
		G1_list.append(G1_broadband[:, :, k])
		G2_list.append(G2_broadband[:, :, k])


	G1 = np.concatenate(G1_list, 0)
	G2 = np.concatenate(G2_list, 0)
	params_values = {}
	params_values['G1'] = G1
	params_values['G2'] = G2
	params_values['Q0'] = 1e-14
<<<<<<< HEAD
	params_values['Q1'] = 0.8 # 0.1 # 0.5 # 1e-8 for u^2, 6e-8 for u^3, 0.5 for 
	params_values['R0'] = 1.5e-12#5e-14 # 3.6e-17#/exp_time**2 #1e-14
=======
	params_values['Q1'] = 0.1 # 0.5 # 1e-8 for u^2, 6e-8 for u^3, 0.5 for
	params_values['R0'] = 5e-14 # 3.6e-17#/exp_time**2 #1e-14
>>>>>>> push test sfr
	params_values['R1'] = 1e-7 # 5e-10
	if model_type == 'reduced':
		G = np.concatenate([G1.real, G1.imag, G2.real, G2.imag], axis=0)
		U, s, V = np.linalg.svd(G, full_matrices=False)
		params_values['U'] = np.matmul(U[:, 0:model_dim], np.diag(np.sqrt(s[0:model_dim])))
		params_values['V'] = np.matmul(np.diag(np.sqrt(s[0:model_dim])), V[0:model_dim, :])

	em_identifier = em.linear_em(params_values, n_pair, model_type=model_type, dim=model_dim)

	for kEM in range(n_EMitr): # run for each system id iteration
		print('********************Itr #{}: running system identification********************'.format(kEM))
		while not os.path.exists(folder+'dataIFS'+str(kEM+1)+'.mat'):
			time.sleep(1)

		dataIFS = sio.loadmat(folder+'dataIFS'+str(kEM+1)+'.mat')

		Ip = dataIFS['dataIFS'+str(kEM+1)]['I'][0, 0]
		u = dataIFS['dataIFS'+str(kEM+1)]['DMcommand'][0, 0]
		up = dataIFS['dataIFS'+str(kEM+1)]['uProbe'][0, 0]
		du = np.zeros(u.shape)
		du[:, 0] = u[:, 0]
		for k in range(1, u.shape[1]):
			du[:, k] = u[:, k] - u[:, k-1]
		Ip_list = []

		for k in range(n_waves):
			Ip_list.append(Ip[:, :, k, :])
		Ip = np.concatenate(Ip_list, 0)

		u1_train = du[0:n_act, :]
		u2_train = du[n_act::, :]
		u1p_train = np.zeros((up.shape[0], 2*n_pair, u1_train.shape[1]+1))
		u2p_train = np.zeros((up.shape[0], 2*n_pair, u1_train.shape[1]+1))
		u1p_train[:, 0::2, :] = up[:, :, :]
		u1p_train[:, 1::2, :] = -up[:, :, :]

		n_step = u1_train.shape[1]
		data_train = {}
		data_train['u1'] = u1_train
		data_train['u2'] = u2_train
		data_train['u1p'] = u1p_train
		data_train['u2p'] = u2p_train
		data_train['I'] = Ip

<<<<<<< HEAD
		# mse_list = em_identifier.train_params(data_train, lr=3e-7, 
								# lr2=3e-3, epoch=2, print_flag=True, params_trainable='jacobian')
		if model_type == 'reduced':
			mse_list = em_identifier.train_params(data_train, lr=3e-6, 
								lr2=3e-3, epoch=2, print_flag=True, params_trainable='all')
		elif model_type == 'normal':
			mse_list = em_identifier.train_params(data_train, lr=3e-7, 
									lr2=3e-3, epoch=2, print_flag=True, params_trainable='all')
		
=======
		mse_list = em_identifier.train_params(data_train, lr=1e-6,
								lr2=1e-2, epoch=3, print_flag=True)

>>>>>>> push test sfr
		for k in range(n_waves):
			G1_broadband[:, :, k] = em_identifier.params_values['G1'][k*n_pix:(k+1)*n_pix, :, 0]
			G2_broadband[:, :, k] = em_identifier.params_values['G2'][k*n_pix:(k+1)*n_pix, :, 0]

		G_broadband = np.concatenate([G1_broadband, G2_broadband], 1)
		sio.savemat(folder+'G_broadband'+str(kEM+1)+'.mat', {'G_broadband': G_broadband})
