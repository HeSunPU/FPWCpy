import scipy.io as sio
import numpy as np
import EMsystemID as em

if __name__ == "__main__":
	# define the EM algorithm system identifier
	n_pair = 2
	n_waves = 5

	modelIFS = sio.loadmat('modelIFS.mat')

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
	params_values['Q1'] = 0.05 # 0.5 # 1e-8 for u^2, 6e-8 for u^3, 0.5 for 
	params_values['R0'] = 3.6e-17#/exp_time**2 #1e-14
	params_values['R1'] = 5e-10

	em_identifier = em.linear_em(params_values, n_pair)

	dataIFS = sio.loadmat('dataIFS.mat')

	Ip = dataIFS['dataIFS']['I'][0, 0]
	u = dataIFS['dataIFS']['DMcommand'][0, 0]
	up = dataIFS['dataIFS']['uProbe'][0, 0]
	du = np.zeros(u.shape)
	du[:, 0] = u[:, 0]
	for k in range(1, u.shape[1]):
		du[:, k] = u[:, k] - u[:, k-1]

	Ip_list = []
	for k in range(n_waves):
		Ip_list.append(Ip[:, :, k, :])
	Ip = np.concatenate(Ip_list, 0)

	u1_train = du[0:952, :]
	u2_train = du[952::, :]
	u1p_train = np.zeros((up.shape[0], 2*n_pair, u1_train.shape[1]))
	u2p_train = np.zeros((up.shape[0], 2*n_pair, u1_train.shape[1]))
	u1p_train[:, 0::2, :] = up[:, :, :]
	u1p_train[:, 1::2, :] = -up[:, :, :]

	data_train = {}
	data_train['u1'] = u1_train
	data_train['u2'] = u2_train
	data_train['u1p'] = u1p_train
	data_train['u2p'] = u2p_train
	data_train['I'] = Ip

	mse_list = em_identifier.train_params(data_train, lr=1e-6, 
							lr2=1e-2, epoch=10, print_flag=True,
							params_trainable='jacobian')

