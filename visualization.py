probe = np.tensordot(np.squeeze(G1), data_train['u1p'], [-1, 0])
probe = np.transpose(probe, [1, 2, 0])
Enp = np.squeeze(np.array(E_true))

Ep = Enp + probe

Ip_pred = np.abs(Ep)**2

Ip = np.transpose(data_train['I'][:, 1:9, :], [1, 2, 0])

observ_noise = np.mean(np.abs(Ip_pred-Ip)**2, -1)

contrast_p = np.mean(Ip, -1)
d_contrast_p = np.mean(np.abs(probe)**2, -1)

observ_noise_sub = observ_noise - params_values['R0']/1 - (params_values['R1']/1 + 4*params_values['Q0']) * contrast_p
dcc = d_contrast_p * contrast_p

ratio = (observ_noise- params_values['R0']/1 - params_values['R1']/1 * contrast_p) / dcc


#%%

dE_true = np.squeeze(np.array(E_true))[1::] - np.squeeze(np.array(E_true))[0:-1]
dE = np.matmul(np.squeeze(G1), data_train['u1']) + np.matmul(np.squeeze(G2), data_train['u2'])
dE = dE.T[0:19, :]
process_noise = np.mean(np.abs(dE - dE_true)**2, 1)


with tf.Session() as sess:
	sess.run(vl.init)
	vl.model.G1_real.load(np.squeeze(vl.params_values['G1']).real)
	vl.model.G1_imag.load(np.squeeze(vl.params_values['G1']).imag)
	vl.model.G2_real.load(np.squeeze(vl.params_values['G2']).real)
	vl.model.G2_imag.load(np.squeeze(vl.params_values['G2']).imag)
	vl.model.q0.load(np.log(vl.params_values['Q0']))
	vl.model.q1.load(np.log(vl.params_values['Q1']))
	vl.model.r0.load(np.log(vl.params_values['R0']))
	vl.model.r1.load(np.log(vl.params_values['R1']))
	for k_params in range(6):
		vl.params_list[k_params].load(vl.params_values['nn_values'][k_params])
	E_next = sess.run(vl.Enp_pred, feed_dict={vl.Enp_old: np.squeeze(np.array(E_true))[0:19],
											vl.u1c: data_train['u1'][:, 0:19].T,
											vl.u2c: data_train['u2'][:, 0:19].T})
	dE_nonlinear = E_next - np.squeeze(np.array(E_true))[0:19]

process_noise_nonlinear2 = np.mean(np.abs(dE_nonlinear - dE_true)**2, 1)


with tf.Session() as sess:
	sess.run(vl.init)
	v_value, r1_value, r2_value = sess.run([v, r1, r2], feed_dict={vl.Enp_old: np.squeeze(np.array(E_true))[0:19]})


delta_G1 = np.matmul(v_value, r1_value)
delta_G2 = np.matmul(v_value, r2_value)

###
benchmark = np.load('vortex_KF_benchmark.npy')
opt_amp = np.load('vortex_KF_opt_amp.npy')
opt_shape = np.load('vortex_KF_opt_shape.npy')
opt_shape_rand = np.load('vortex_KF_opt_shape_random.npy')

time_1 = np.cumsum(np.concatenate([np.zeros(1), np.ones(19)*8]))
time_2 = np.cumsum(np.concatenate([np.zeros(1), np.ones(39)*4]))


matplotlib.rcParams.update({'font.size': 12})

plt.figure(1, figsize=(6.8, 5))
plt.figure(1), plt.semilogy(time_1, benchmark.item()['contrast'], 'r-o')
plt.figure(1), plt.semilogy(time_1, opt_amp.item()['contrast'], 'g-x')
plt.figure(1), plt.semilogy(time_2, opt_shape.item()['contrast'], 'b-^')
plt.figure(1), plt.semilogy(time_2, opt_shape_rand.item()['contrast'], 'm-s')
plt.figure(1), plt.semilogy(time1[0:25], SNR1.item()['contrast'][0:25], 'k->')


plt.xlabel('time (seconds)', fontsize=16)
plt.ylabel('contrast', fontsize=16)
plt.legend(['benchmark sinc', 'opt amp sinc', 'opt shape sinc init', 'opt shape rand init', 'adaptive time SNR 1'], fontsize=14)
###
probe2D1 = np.zeros((41, 49), dtype=np.complex)
probe2D2 = np.zeros((41, 49), dtype=np.complex)

probe2D1[dh_ind1, dh_ind2] = np.matmul(np.squeeze(G1), opt_shape_rand.item()['u1p'][:, 0, 0])
probe2D2[dh_ind1, dh_ind2] = np.matmul(np.squeeze(G1), opt_shape_rand.item()['u1p'][:, 2, 0])
angle_diff = np.abs(np.angle(probe2D2)-np.angle(probe2D1))%np.pi*180/np.pi
angle_diff[angle_diff==0] = np.nan

plt.figure(2), plt.imshow(np.log10(np.abs(probe2D1)**2))
# plt.axis('off')
plt.clim([-10, -5])
plt.colorbar()

plt.figure(3), plt.imshow(np.log10(np.abs(probe2D2)**2))
# plt.axis('off')
plt.clim([-10, -5])
plt.colorbar()

plt.figure(4), plt.imshow(angle_diff)
# plt.axis('off')
plt.clim([0, 180])
plt.colorbar()

k = 0.75e-10
SNR = 1 / (1/40 + 5e-10/(8*k) + 0.5 * np.sqrt(0.05*(3.6e-17/k**2 + 5e-10/k)))
print('SNR: {}'.format(SNR))


###
SNR05= np.load('vortex_KF_opt_shape_SNR05.npy')
SNR1 = np.load('vortex_KF_opt_shape_SNR1.npy')
SNR3 = np.load('vortex_KF_opt_shape_SNR3.npy')
SNR10 = np.load('vortex_KF_opt_shape_SNR10.npy')
SNR10_amp = np.load('vortex_KF_opt_amp_SNR10.npy')
SNR3_amp = np.load('vortex_KF_opt_amp_SNR3.npy')
SNR1_amp = np.load('vortex_KF_opt_amp_SNR1.npy')
SNR05_amp = np.load('vortex_KF_opt_amp_SNR05.npy')

opt_shape = np.load('vortex_KF_opt_shape.npy')
opt_shape01 = np.load('vortex_KF_opt_shape_time01.npy')





time05 = np.cumsum(np.concatenate([np.zeros(1), 4*SNR05.item()['time'][0:-1]]))
time1 = np.cumsum(np.concatenate([np.zeros(1), 4*SNR1.item()['time'][0:99]]))
time3 = np.cumsum(np.concatenate([np.zeros(1), 4*SNR3.item()['time'][0:39]]))
time10 = np.cumsum(np.concatenate([np.zeros(1), 4*SNR10.item()['time'][0:39]]))

time10_amp = np.cumsum(np.concatenate([np.zeros(1), 8*SNR10_amp.item()['time'][0:19]]))
time3_amp = np.cumsum(np.concatenate([np.zeros(1), 8*SNR3_amp.item()['time'][0:-1]]))
time1_amp = np.cumsum(np.concatenate([np.zeros(1), 8*SNR1_amp.item()['time'][0:-1]]))
time05_amp = np.cumsum(np.concatenate([np.zeros(1), 8*SNR05_amp.item()['time'][0:-1]]))

timesp1 = np.cumsum(np.concatenate([np.zeros(1), 4*opt_shape.item()['time'][0:-1]]))
timesp01 = np.cumsum(np.concatenate([np.zeros(1), 4*opt_shape01.item()['time'][0:-1]]))


matplotlib.rcParams.update({'font.size': 12})

plt.figure(5, figsize=(6.8, 5))
plt.figure(5), plt.loglog(time05, SNR05.item()['contrast'], color='cyan', marker='o')
plt.figure(5), plt.loglog(time1, SNR1.item()['contrast'], color='black', marker='>')
plt.figure(5), plt.loglog(time3, SNR3.item()['contrast'], color='orange', marker='^')
plt.figure(5), plt.loglog(time10, SNR10.item()['contrast'], color='brown', marker='s')
# plt.figure(5), plt.loglog(time10_amp, SNR10_amp.item()['contrast'], 'm--s')
# plt.figure(5), plt.loglog(time3_amp, SNR3_amp.item()['contrast'], 'b--^')
# plt.figure(5), plt.loglog(time1_amp, SNR1_amp.item()['contrast'], 'g--x')
# plt.figure(5), plt.loglog(time05_amp, SNR05_amp.item()['contrast'], 'r--o')

# plt.figure(5), plt.loglog(timesp1, opt_shape.item()['contrast'], 'b-^')
# plt.figure(5), plt.loglog(timesp01, opt_shape01.item()['contrast'], 'k-.')

plt.xlim([1e-1, 1e3])
plt.ylim([3e-11, 1e-6])

plt.xlabel('time (seconds)', fontsize=16)
plt.ylabel('contrast', fontsize=16)

plt.legend(['adaptive time SNR 1/2', 'adaptive time SNR 1', 'adaptive time SNR 3', 'adaptive time SNR 10'], fontsize=14)