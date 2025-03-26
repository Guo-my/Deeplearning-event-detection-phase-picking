import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.signal import stft
import os
from matplotlib.ticker import FormatStrFormatter
# from Utils import calculate_snr, normalize, mean_normalization
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

data = np.load('./comparing/macro_test_set_raw.npy').reshape(-1, 6000, 3)
dt = np.load('./comparing/macro_test_set_d.npy').squeeze().reshape(-1, 6000)
pt = np.load('./comparing/macro_test_set_p.npy').squeeze().reshape(-1, 6000)
st = np.load('./comparing/macro_test_set_s.npy').squeeze().reshape(-1, 6000)
pre_d = np.load('./our_detect_model_result/macro_detect_d_result.npy').squeeze()
pre_p = np.load('./our_detect_model_result/macro_detect_p_result.npy').squeeze()
pre_s = np.load('./our_detect_model_result/macro_detect_s_result.npy').squeeze()

# data = np.load('./comparing/aug_test_set_raw.npy').reshape(-1, 6000, 3)
# dt = np.load('./comparing/aug_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/aug_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/aug_test_set_s.npy').squeeze().reshape(-1, 6000)
# pre_d = np.load('./our_detect_model_result/high_detect_d_result.npy').squeeze()
# pre_p = np.load('./our_detect_model_result/high_detect_p_result.npy').squeeze()
# pre_s = np.load('./our_detect_model_result/high_detect_s_result.npy').squeeze()
#
# data = np.load('./comparing/low_loc_test_set_raw.npy').reshape(-1, 6000, 3)
# dt = np.load('./comparing/low_loc_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/low_loc_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/low_loc_test_set_s.npy').squeeze().reshape(-1, 6000)

k_list = [2, 3, 4, 5]

# save_path = './comparing/test'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# np.save(os.path.join(save_path, 'low_data.npy'), data[k_list])
# np.save(os.path.join(save_path, 'low_d.npy'), dt[k_list])
# np.save(os.path.join(save_path, 'low_p.npy'), pt[k_list])
# np.save(os.path.join(save_path, 'low_s.npy'), st[k_list])
# np.save(os.path.join(save_path, 'low_azi.npy'), azi[k_list])
# np.save(os.path.join(save_path, 'low_dist.npy'), dist[k_list])

def noise(noisedata):
    length = len(noisedata)
    num = 6000 // length
    noisesignal = np.empty(6000)
    for i in range(num):
        noisesignal[i*length:i*length + length] = noisedata[:]
    noisesignal[num*length:] = noisedata[:6000-num*length]
    return noisesignal

def calculate_snr(signal, noise):
    signal_power = np.sum(np.square(signal))+ 1e-8
    noise_power = np.sum(np.square(noise))+ 1e-8
    snr = 10 * np.log10(signal_power/ noise_power)
    return snr

# SNR = []
# for i in range(data.shape[0]):
#     data_ = data[i, :, 0]
#     if np.where(pt[i, :] == 1)[0].size != 1:
#         continue
#     if np.where(st[i, :] == 1)[0].size != 1:
#         continue
#     p_point = int(np.where(pt[i, :] == 1)[0])
#     s_point = int(np.where(st[i, :] == 1)[0])
#     sp = int((s_point - p_point) * 1.4 + s_point)
#     s = data_[p_point:sp]
#     # print(s.shape)
#     # s_95 = np.percentile(np.abs(s), 95)
#     # s = [min(x, s_95) for x in s]
#     n = data_[:p_point]
#     n = noise(n)
#     # n_95 = np.percentile(np.abs(n), 95)
#     # n = [min(x, n_95) for x in n]
#     # data = predata_filter[i, :]
#     # data[ptime[i]:end[i]] = 0
#     SNR.append(calculate_snr(s, n))
#
# k_list = [index for index, value in enumerate(SNR) if value < 0]


threshold = 0.3
x_data = np.linspace(0, 60, 6000)
for k in range(pre_d.shape[0]):
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.2)

    axs[0].plot(x_data, data[k, :, 0], 'k', label='E component')
    axs[0].set_xticklabels([])
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].set_ylabel('Amplitude')
    if data[k, :, 0].all() != 0:
        if len(np.where(pt[k, :] == 1)[0]) == 1 & len(np.where(st[k, :] == 1)[0]) == 1:
            indexes_p = float(np.where(pt[k, :] == 1)[0]) / 100
            indexes_s = float(np.where(st[k, :] == 1)[0]) / 100
            axs[0].axvline(indexes_p, color='#00FFFF', linewidth=2, label='Manual p-arrival')
            axs[0].axvline(indexes_s, color='#FF00FF', linewidth=2, label='Manual s-arrival')
        elif len(np.where(pt[k, :] == 1)[0]) == 2 & len(np.where(st[k, :] == 1)[0]) == 2:
            indexes_p = float(np.where(pt[k, :] == 1)[0][0]) / 100
            indexes_second_p = float(np.where(pt[k, :] == 1)[0][1]) / 100
            indexes_s = float(np.where(st[k, :] == 1)[0][0]) / 100
            indexes_second_s = float(np.where(st[k, :] == 1)[0][1]) / 100
            axs[0].axvline(indexes_p, color='#00FFFF', linewidth=2, label='Manual p-arrival')
            axs[0].axvline(indexes_s, color='#FF00FF', linewidth=2, label='Manual s-arrival')
            axs[0].axvline(indexes_second_p, color='#00FFFF', linewidth=2)
            axs[0].axvline(indexes_second_s, color='#FF00FF', linewidth=2)

    axs[0].legend(loc='upper right')

    axs[1].plot(x_data, data[k, :, 1], 'k', label='N component')
    axs[1].set_xticklabels([])
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].set_ylabel('Amplitude')
    if data[k, :, 1].all() != 0:
        if len(np.where(pt[k, :] == 1)[0]) == 1 & len(np.where(st[k, :] == 1)[0]) == 1:
            indexes_p = float(np.where(pt[k, :] == 1)[0]) / 100
            indexes_s = float(np.where(st[k, :] == 1)[0]) / 100
            axs[1].axvline(indexes_p, color='#00FFFF', linewidth=2, label='Manual p-arrival')
            axs[1].axvline(indexes_s, color='#FF00FF', linewidth=2, label='Manual s-arrival')
        elif len(np.where(pt[k, :] == 1)[0]) == 2 & len(np.where(st[k, :] == 1)[0]) == 2:
            indexes_p = float(np.where(pt[k, :] == 1)[0][0]) / 100
            indexes_second_p = float(np.where(pt[k, :] == 1)[0][1]) / 100
            indexes_s = float(np.where(st[k, :] == 1)[0][0]) / 100
            indexes_second_s = float(np.where(st[k, :] == 1)[0][1]) / 100
            axs[1].axvline(indexes_p, color='#00FFFF', linewidth=2, label='Manual p-arrival')
            axs[1].axvline(indexes_s, color='#FF00FF', linewidth=2, label='Manual s-arrival')
            axs[1].axvline(indexes_second_p, color='#00FFFF', linewidth=2)
            axs[1].axvline(indexes_second_s, color='#FF00FF', linewidth=2)
    axs[1].legend(loc='upper right')

    axs[2].plot(x_data, data[k, :, 2], 'k', label='Z component')
    axs[2].set_xticklabels([])
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_ylabel('Amplitude')
    if data[k, :, 2].all() != 0:
        if len(np.where(pt[k, :] == 1)[0]) == 1 & len(np.where(st[k, :] == 1)[0]) == 1:
            max_value_p = np.max(pre_p[k, :] * (pre_p[k, :] > threshold))
            if max_value_p == 0:
                indexes_p = None
            else:
                indexes_p = float(np.where(pre_p[k, :] == max_value_p)[0])/100
                axs[2].axvline(indexes_p, color='#665fd1', linewidth=2, label='Predicted p')
            max_value_s = np.max(pre_s[k, :] * (pre_s[k, :] > threshold))
            if max_value_s == 0:
                indexes_s = None
            else:
                indexes_s = float(np.where(pre_s[k, :] == max_value_s)[0])/100
                axs[2].axvline(indexes_s, color='#c44240', linewidth=2, label='Predicted s')
        elif len(np.where(pt[k, :] == 1)[0]) == 2 & len(np.where(st[k, :] == 1)[0]) == 2:
            indexes_second_p = int(np.where(pt[k, :] == 1)[0][1]) - 1
            max_value_p = np.max(pre_p[k, :indexes_second_p] * (pre_p[k, :indexes_second_p] > threshold))
            if max_value_p == 0:
                indexes_p = None
            else:
                indexes_p = float(np.where(pre_p[k, :indexes_second_p] == max_value_p)[0]) / 100
                axs[2].axvline(indexes_p, color='#665fd1', linewidth=2, label='Predicted p')
            second_max_value_p = np.max(pre_p[k, indexes_second_p:] * (pre_p[k, indexes_second_p:] > threshold))
            if second_max_value_p == 0:
                second_indexes_p = None
            else:
                second_indexes_p = float(np.where(pre_p[k, indexes_second_p:] == second_max_value_p)[0] + indexes_second_p) / 100
                axs[2].axvline(second_indexes_p, color='#665fd1', linewidth=2)
            max_value_s = np.max(pre_s[k, :indexes_second_p] * (pre_s[k, :indexes_second_p] > threshold))
            if max_value_s == 0:
                indexes_s = None
            else:
                indexes_s = float(np.where(pre_s[k, :indexes_second_p] == max_value_s)[0]) / 100
                axs[2].axvline(indexes_s, color='#c44240', linewidth=2, label='Predicted s')
            second_max_value_s = np.max(pre_s[k, indexes_second_p:] * (pre_s[k, indexes_second_p:] > threshold))
            if second_max_value_s == 0:
                second_indexes_s = None
            else:
                second_indexes_s = float(np.where(pre_s[k, indexes_second_p:] == second_max_value_s)[0] + indexes_second_p) / 100
                axs[2].axvline(second_indexes_s, color='#c44240', linewidth=2)
    axs[2].legend(loc='upper right')

    axs[3].plot(x_data, pre_d[k, :], '#65ab7c', linestyle='--', linewidth=2, label='Predicted event')
    axs[3].plot(x_data, pre_p[k, :], '#665fd1', linestyle='--', linewidth=2, label='Predicted p')
    axs[3].plot(x_data, pre_s[k, :], '#c44240', linestyle='--', linewidth=2, label='Predicted s')
    axs[3].set_ylabel('Probability')
    axs[3].set_ylim(0, 1.1)
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')

    labels = ['(I)', '(II)', '(III)', '(IV)', '(V)', '(VI)', '(VII)', '(VIII)', '(IX)', '(X)']
    for ax, label in zip(axs.flat, labels):
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1))
    plt.savefig(f'./figure_plot/pick_plot/comparing/low/macro_pick_comparing_{k}.jpg', format='jpg', transparent=True, dpi=600)
    plt.show()
    # print(k)
    # inp = input("Press a key to plot the next waveform!")
    # if inp == "r":
    #     continue