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
G1 = np.load('vortex_compact_Jacobian1.npy')
G2 = np.load('vortex_compact_Jacobian2.npy')
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


bpe_contrast = benchmark.item()['contrast']
opt_amp_contrast = opt_amp.item()['contrast']
opt_shape_contrast = opt_shape.item()['contrast']
opt_shape_rand_contrast = opt_shape_rand.item()['contrast']
SNR05_contrast = SNR05.item()['contrast']
SNR1_contrast = SNR1.item()['contrast']
SNR3_contrast = SNR3.item()['contrast']
SNR10_contrast = SNR10.item()['contrast']



sio.savemat('bpe_contrast.mat', {'bpe_contrast': bpe_contrast})
sio.savemat('opt_amp_contrast.mat', {'opt_amp_contrast': opt_amp_contrast})
sio.savemat('opt_shape_contrast.mat', {'opt_shape_contrast': opt_shape_contrast})
sio.savemat('opt_shape_rand_contrast.mat', {'opt_shape_rand_contrast': opt_shape_rand_contrast})
sio.savemat('SNR05_contrast.mat', {'SNR05_contrast': SNR05_contrast})
sio.savemat('SNR1_contrast.mat', {'SNR1_contrast': SNR1_contrast})
sio.savemat('SNR3_contrast.mat', {'SNR3_contrast': SNR3_contrast})
sio.savemat('SNR10_contrast.mat', {'SNR10_contrast': SNR10_contrast})

sio.savemat('time_1.mat', {'time_1': time_1})
sio.savemat('time_2.mat', {'time_2': time_2})
sio.savemat('time05.mat', {'time05': time05})
sio.savemat('time1.mat', {'time1': time1})
sio.savemat('time3.mat', {'time3': time3})
sio.savemat('time10.mat', {'time10': time10})

sio.savemat('y_coord.mat', {'y_coord': y_coord})
sio.savemat('x_coord.mat', {'x_coord': x_coord})

probe2D1_opt_shape_rand = probe2D1
probe2D2_opt_shape_rand = probe2D2
angle_diff_opt_shape_rand = angle_diff
sio.savemat('probe2D1_opt_shape_rand.mat', {'probe2D1_opt_shape_rand': probe2D1_opt_shape_rand})
sio.savemat('probe2D2_opt_shape_rand.mat', {'probe2D2_opt_shape_rand': probe2D2_opt_shape_rand})
sio.savemat('angle_diff_opt_shape_rand.mat', {'angle_diff_opt_shape_rand': angle_diff_opt_shape_rand})

probe2D1_opt_shape = probe2D1
probe2D2_opt_shape = probe2D2
angle_diff_opt_shape = angle_diff
sio.savemat('probe2D1_opt_shape.mat', {'probe2D1_opt_shape': probe2D1_opt_shape})
sio.savemat('probe2D2_opt_shape.mat', {'probe2D2_opt_shape': probe2D2_opt_shape})
sio.savemat('angle_diff_opt_shape.mat', {'angle_diff_opt_shape': angle_diff_opt_shape})


probe2D1_benchmark = probe2D1
probe2D2_benchmark = probe2D2
probe2D3_benchmark = probe2D3
probe2D4_benchmark = probe2D4
angle_diff1_benchmark = angle_diff1
angle_diff2_benchmark = angle_diff2

sio.savemat('probe2D1_benchmark.mat', {'probe2D1_benchmark': probe2D1_benchmark})
sio.savemat('probe2D2_benchmark.mat', {'probe2D2_benchmark': probe2D2_benchmark})
sio.savemat('probe2D3_benchmark.mat', {'probe2D3_benchmark': probe2D3_benchmark})
sio.savemat('probe2D4_benchmark.mat', {'probe2D4_benchmark': probe2D4_benchmark})
sio.savemat('angle_diff1_benchmark.mat', {'angle_diff1_benchmark': angle_diff1_benchmark})
sio.savemat('angle_diff2_benchmark.mat', {'angle_diff2_benchmark': angle_diff2_benchmark})


plt.xlabel('time (seconds)', fontsize=16)
plt.ylabel('contrast', fontsize=16)
plt.legend(['benchmark sinc', 'opt amp sinc', 'opt shape sinc init', 'opt shape rand init', 'adaptive time SNR 1'], fontsize=14)
###
probe2D1 = np.zeros((41, 49), dtype=np.complex)
probe2D2 = np.zeros((41, 49), dtype=np.complex)
probe2D3 = np.zeros((41, 49), dtype=np.complex)
probe2D4 = np.zeros((41, 49), dtype=np.complex)


probe2D1[dh_ind1, dh_ind2] = np.matmul(np.squeeze(G1), opt_amp.item()['u1p'][:, 0, 0])
probe2D2[dh_ind1, dh_ind2] = np.matmul(np.squeeze(G1), opt_amp.item()['u1p'][:, 4, 0])
probe2D3[dh_ind1, dh_ind2] = np.matmul(np.squeeze(G1), opt_amp.item()['u1p'][:, 2, 0])
probe2D4[dh_ind1, dh_ind2] = np.matmul(np.squeeze(G1), opt_amp.item()['u1p'][:, 6, 0])

angle_diff1 = np.abs(np.angle(probe2D2)-np.angle(probe2D1))%np.pi*180/np.pi
angle_diff2 = np.abs(np.angle(probe2D4)-np.angle(probe2D3))%np.pi*180/np.pi
angle_diff1[angle_diff1==0] = np.nan
angle_diff2[angle_diff1==0] = np.nan


y_coord = np.arange(-20, 21, 1) * model.camera_binXi * model.camera_pitch / (model.focalLength * wavelength / model.SPwidth)
x_coord = np.arange(-24, 25, 1) * model.camera_binXi * model.camera_pitch / (model.focalLength * wavelength / model.SPwidth)

plt.figure(2), plt.imshow(np.log10(np.abs(probe2D1)**2), extent = [y_coord[0], y_coord[-1], x_coord[0], x_coord[-1]])
# plt.axis('off')
plt.clim([-10, -5])
plt.colorbar()

plt.figure(3), plt.imshow(np.log10(np.abs(probe2D2)**2), extent = [y_coord[0], y_coord[-1], x_coord[0], x_coord[-1]])
# plt.axis('off')
plt.clim([-10, -5])
plt.colorbar()

plt.figure(4), plt.imshow(angle_diff, extent = [y_coord[0], y_coord[-1], x_coord[0], x_coord[-1]])
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


#%%
import scipy.io as sio
import scipy as sp
import numpy as np
import VORTEX_model as vortex
wavelength = 635e-9 * np.ones(1)
DM1gain = 5.06e-9 * np.ones((34, 34))
DM2gain = 6.27e-9 * np.ones((34, 34))
coronagraph_type = 'vortex'#'splc'#
model_perfect = vortex.Optical_Model(wavelength, DM1gain, DM2gain, wfe=False)
u1 = np.zeros((34, 34))
u2 = np.zeros((34, 34))
Ef = model_perfect.Propagate(u1, u2)
Ep = model_perfect.Propagate(u1, u2, to_pupil=True)


I0 = np.abs(model.Propagate(np.zeros(u1.shape), np.zeros(u2.shape)))**2
Iend = np.abs(model.Propagate(u1, u2))**2
I0 = I0[:, :, 0]
Iend = Iend[:, :, 0]

sio.savemat('I0.mat', {'I0': I0})
sio.savemat('Iend.mat', {'Iend': Iend})
