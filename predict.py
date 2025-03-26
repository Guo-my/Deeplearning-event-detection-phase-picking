import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from .model.Detect_pick_model import Generalist_without_denoise
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from Error import calculate_error

def noise(noisedata):
    length = len(noisedata)
    num = 6000 // length
    noisesignal = np.empty(6000)
    for i in range(num):
        noisesignal[i*length:i*length + length] = noisedata[:]
    noisesignal[num*length:] = noisedata[:6000-num*length]
    return noisesignal


def calculate_snr(signal, noise):
    'Calculate the SNR'

    signal_power = np.sum(np.square(signal))+ 1e-8
    noise_power = np.sum(np.square(noise))+ 1e-8
    snr = 10 * np.log10(signal_power/ noise_power)
    return snr


def normalize(data, mode='std'):
    'Normalize waveforms in each batch'

    data -= np.mean(data, axis=2, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=2, keepdims=True)
        assert (max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=2, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    return data


def batch(data):
    'Divide the data into batches.'

    batch_size = 128
    batch_data = []
    batch_num = data.shape[0]//batch_size + 1
    for i in range(batch_num):
        batch_data.append(data[i*batch_size:(i+1)*batch_size])
    return batch_data


# file_name = "../STEAD/merged/merge.hdf5"
# csv_file = "../STEAD/merged/merge.csv"
# test = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=True, mode='test')
# clean_data = np.load('./comparing/test_set_raw.npy').reshape(-1, 6000, 3).transpose(0, 2, 1)
# data = np.load('./comparing/aug_test_set_raw.npy')
# label_dt = np.load('./comparing/aug_test_set_d.npy')
# label_pt = np.load('./comparing/aug_test_set_p.npy')
# label_st = np.load('./comparing/aug_test_set_s.npy')
# dt = np.load('./comparing/aug_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/aug_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/aug_test_set_s.npy').squeeze().reshape(-1, 6000)
# label_dist = np.load('./comparing/dist_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/dist_test_set_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/test_set_azimuth.npy').squeeze().reshape(-1, 2)
# data = np.load('..//new_file/MODEL_TRAIN/label_test_new.npy')
# data = normalize(data)
# print(label_azi.shape)

# data = np.load('./comparing/high_loc_test_set_raw.npy')
# label_dt = np.load('./comparing/high_loc_test_set_d.npy')
# label_pt = np.load('./comparing/high_loc_test_set_p.npy')
# label_st = np.load('./comparing/high_loc_test_set_s.npy')
# dt = np.load('./comparing/high_loc_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/high_loc_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/high_loc_test_set_s.npy').squeeze().reshape(-1, 6000)
# label_dist = np.load('./comparing/high_loc_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/high_loc_test_set_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/high_loc_test_set_azimuth.npy').squeeze().reshape(-1, 2)

# data = np.load('./comparing/low_loc_test_set_raw.npy')
# label_dt = np.load('./comparing/low_loc_test_set_d.npy')
# label_pt = np.load('./comparing/low_loc_test_set_p.npy')
# label_st = np.load('./comparing/low_loc_test_set_s.npy')
# dt = np.load('./comparing/low_loc_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/low_loc_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/low_loc_test_set_s.npy').squeeze().reshape(-1, 6000)
# label_dist = np.load('./comparing/low_loc_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/low_loc_test_set_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/low_loc_test_set_azimuth.npy').squeeze().reshape(-1, 2)

# data = np.load('./comparing/macro_test_set_raw.npy')
# label_dt = np.load('./comparing/macro_test_set_d.npy')
# label_pt = np.load('./comparing/macro_test_set_p.npy')
# label_st = np.load('./comparing/macro_test_set_s.npy')
# dt = np.load('./comparing/macro_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/macro_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/macro_test_set_s.npy').squeeze().reshape(-1, 6000)
# label_dist = np.load('./comparing/macro_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/macro_test_set_p_travel.npy').squeeze().reshape(-1)

# data = np.load('./comparing/low_snr_test_set_raw.npy')
# label_dt = np.load('./comparing/low_snr_test_set_d.npy')
# label_pt = np.load('./comparing/low_snr_test_set_p.npy')
# label_st = np.load('./comparing/low_snr_test_set_s.npy')
# dt = np.load('./comparing/low_snr_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/low_snr_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/low_snr_test_set_s.npy').squeeze().reshape(-1, 6000)
# label_dist = np.load('./comparing/low_snr_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/low_snr_test_set_p_travel.npy').squeeze().reshape(-1)

# The shape of data: [batch_size, channel, samples]
# real_data = np.load('./JAPAN_test/waveform.npy')
real_data_E = np.load('./1JAPAN_test/prompt_E_wave.npy')
real_data_N = np.load('./1JAPAN_test/prompt_N_wave.npy')
real_data_Z = np.load('./1JAPAN_test/prompt_Z_wave.npy')
real_data = np.stack((real_data_E, real_data_N, real_data_Z), axis=1)
print(real_data.shape)
real_data = normalize(real_data)
print(real_data.shape)
batch_real_data = batch(real_data)
print(len(batch_real_data))

# Loading the model onto the GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Generalist_without_denoise()
model = model.to(device)

model_file_path = './Project2/tiaoshi_model/detect_test_netpar_epoch_50.pth'

# Loading the parameters
if os.path.isfile(model_file_path):
    model.load_state_dict(torch.load(model_file_path))
    print("load parameters successfully")
else:
    print("no parameters")

# Ready to predict!
model.eval()
pre_d = []
pre_p = []
pre_s = []
pre_dist = []
pre_p_travel = []
pre_azi = []
true_d = []
true_p = []
true_s = []
test_bar = tqdm(range(1), colour="white")
with torch.no_grad():
    for i in test_bar:
        # test_x = torch.tensor(data[i], dtype=torch.float32).permute(0, 2, 1)
        # test_d = torch.tensor(label_dt[i], dtype=torch.float32).permute(0, 2, 1)
        # test_p = torch.tensor(label_pt[i], dtype=torch.float32).permute(0, 2, 1)
        # test_s = torch.tensor(label_st[i], dtype=torch.float32).permute(0, 2, 1)
        test_x = torch.tensor(batch_real_data[i], dtype=torch.float32)
        # test_x_ssq = ssq_function(test_x)
        #
        # test_x_ssq = test_x_ssq.to(device)
        test_x = test_x.to(device)
        # test_d = test_d.to(device)
        # test_p = test_p.to(device)
        # test_s = test_s.to(device)
        output_d, output_p, output_s = model(test_x)
        # output = model(test_x)
        # output_dist, output_p_travel = model(test_x)
        # output_dist = model(test_x)
        # output_azi = model(test_x)
        # output_d = output[:, 0, :][:, None, :]
        # output_p = output[:, 1, :][:, None, :]
        # output_s = output[:, 2, :][:, None, :]
        # fig, ax = plt.subplots(11, 1)
        # ax[0].plot(test_x[65, 0, :].detach().cpu())
        # ax[0].set_xticks([])
        # ax[1].plot(output_d[65, 0, :].detach().cpu())
        # ax[1].set_xticks([])
        # ax[2].plot(output_p[65, 0, :].detach().cpu())
        # ax[2].set_xticks([])
        # ax[3].plot(output_s[65, 0, :].detach().cpu())
        # ax[3].set_xticks([])
        # ax[4].plot(prob[65, 0, :].detach().cpu())
        # ax[4].set_xticks([])
        # plt.show()
        # inp = input("Press a key to plot the next waveform!")
        # if inp == "r":
        #     continue
        # output_d, output_p, output_s = model(test_x)
        # print(output_d)
        # true_d.append(one_batch_d)
        # true_p.append(one_batch_p)
        # true_s.append(one_batch_s)
        pre_d.append(output_d.cpu().numpy())
        pre_p.append(output_p.cpu().numpy())
        pre_s.append(output_s.cpu().numpy())
        # pre_dist.append(output_dist.cpu().numpy())
        # pre_p_travel.append(output_p_travel.cpu().numpy())
        # pre_azi.append(output_azi.cpu().numpy())

# pre_d_result = pre_d[0]
# pre_p_result = pre_p[0]
# pre_s_result = pre_s[0]
pre_d_result = np.concatenate(pre_d, axis=0)
pre_p_result = np.concatenate(pre_p, axis=0)
pre_s_result = np.concatenate(pre_s, axis=0)

# ---------------------------------------------------------------------------------------------------------
# You can save the prediction results here!
# pre_dist_result = np.concatenate(pre_dist, axis=0).squeeze()
# pre_p_travel_result = np.concatenate(pre_p_travel, axis=0).squeeze()
# pre_azimuth = np.concatenate(pre_azi, axis=0).squeeze()

# save_path = './japan_test_our_detect_model_result'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# np.save(os.path.join(save_path, 'detect_d_result.npy'), pre_d_result)
# np.save(os.path.join(save_path, 'detect_p_result.npy'), pre_p_result)
# np.save(os.path.join(save_path, 'detect_s_result.npy'), pre_s_result)

# save_path = './3japan_test_our_dist_nn_result'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# np.save(os.path.join(save_path, 'loc_dist_result.npy'), pre_dist_result)
# np.save(os.path.join(save_path, 'loc_p_travel_result.npy'), pre_p_travel_result)

# save_path = './japan_test_our_azi_model_result'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# np.save(os.path.join(save_path, 'loc_azi_result.npy'), pre_azimuth)

# ---------------------------------------------------------------------------------------------------------
# Calculate the metrics of event detect and phase picking
mean_error_p, sigma_p, mae_p, mape_p, error_p_plot = calculate_error(pt, pre_p_result, 0.3)
mean_error_s, sigma_s, mae_s, mape_s, error_s_plot = calculate_error(st, pre_s_result, 0.3)
print(mean_error_p, sigma_p, mae_p, mape_p)
print(mean_error_s, sigma_s, mae_s, mape_s)
save_path = './just_for_test_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(save_path, 'sigma_p_plot.npy'), error_p_plot)
np.save(os.path.join(save_path, 'sigma_s_plot.npy'), error_s_plot)
print(error_p_plot)

# bin_counts1, bin_edges1 = np.histogram(error_p_plot, bins=300)
# plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
#         color="#4169E1", label='Mean=%.4f \n Std=%.4f \n MAE=%.4f \n MAPE=%.4f' % (mean_error_p, sigma_p, mae_p, mape_p))
# plt.legend(fontsize=10)
# plt.xlim(-0.5, 0.5)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Data Count')
# plt.show()
#
# bin_counts1, bin_edges1 = np.histogram(error_s_plot, bins=900)
# plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
#         color="#FF6347", label='Mean=%.4f \n Std=%.4f \n MAE=%.4f \n MAPE=%.4f' % (mean_error_s, sigma_s, mae_s, mape_s))
# plt.legend(fontsize=10)
# plt.xlim(-0.5, 0.5)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Number of Picks')
# plt.show()
#
# min_edge = min(np.min(error_p_plot), np.min(error_s_plot))
# max_edge = max(np.max(error_p_plot), np.max(error_s_plot))
# num = int((max_edge - min_edge) / 0.03) + 1
#
# bin_edges = np.linspace(min_edge, max_edge, num)
# bin_counts1, _ = np.histogram(error_p_plot, bins=bin_edges)
# bin_counts2, _ = np.histogram(error_s_plot, bins=bin_edges)
#
# plt.bar(bin_edges[:-1], bin_counts1, width=np.diff(bin_edges), edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.4f \n Std=%.4f' % (mae_p, sigma_p))
# plt.legend(fontsize=10)
# plt.xlim(-0.5, 0.5)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Number of Picks')
# plt.savefig(f'./figure_plot/pick_plot/EQT_p_time_residual.png', format='png', transparent=True, dpi=600)
# plt.show()
#
# plt.bar(bin_edges[:-1], bin_counts2, width=np.diff(bin_edges), edgecolor="white", align="edge",
#         color="#FF6347", label='MAE=%.4f \n Std=%.4f' % (mae_s, sigma_s))
# plt.legend(fontsize=10)
# plt.xlim(-0.5, 0.5)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Number of Picks')
# plt.savefig(f'./figure_plot/pick_plot/EQT_s_time_residual.png', format='png', transparent=True, dpi=600)
# plt.show()

# ---------------------------------------------------------------------------------------------------------
# Plot the distribution of the estimation epicenter distance and p_travel time
# plot_dist = pre_dist_result - label_dist
# plot_p_travel = pre_p_travel_result - label_p_travel
#
# mean_error_dist = np.mean(plot_dist)
# mae_dist = np.mean(np.abs(plot_dist))
# sigma_dist = np.std(plot_dist)
#
# mean_error_p_travel = np.mean(plot_p_travel)
# mae_p_travel = np.mean(np.abs(plot_p_travel))
# sigma_p_travel = np.std(plot_p_travel)
#
# bin_counts1, bin_edges1 = np.histogram(plot_dist, bins=30)
# plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.4f \n Std=%.4f' % (mae_dist, sigma_dist))
# plt.legend(fontsize=10)
# # plt.xlim(-0.5, 0.5)
# plt.xlabel('Dist residuals (km)')
# plt.ylabel('Frequency')
# # plt.savefig(f'./figure_plot/dist_plot/mousavi_dist_residual.png', format='png', transparent=True, dpi=600)
# plt.show()
#
# bin_counts1, bin_edges1 = np.histogram(plot_p_travel, bins=100)
# plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
#         color="#FF6347", label='MAE=%.4f \n Std=%.4f' % (mae_p_travel, sigma_p_travel))
# plt.legend(fontsize=10)
# plt.xlim(-10, 10)
# plt.xlabel('Time residuals (s)')
# plt.ylabel('Frequency')
# # plt.savefig(f'./figure_plot/dist_plot/mousavi_time_residual.png', format='png', transparent=True, dpi=600)
# plt.show()

# ---------------------------------------------------------------------------------------------------------
# Plot the distribution scatter of the estimation epicenter distance and p_travel time
# dist_r_squared = r2_score(label_dist, pre_dist_result)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_dist, pre_dist_result, color='#4169E1', alpha=0.5)
# plt.plot([0, 110], [0, 110], color='white', linestyle='--', linewidth=3)
# plt.text(0, 100, f'R² = {dist_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
# plt.xlabel('True Distance (km)', fontsize=15)
# plt.ylabel('Predicted Distance (km)', fontsize=15)
# # plt.legend()
# plt.grid()
# # plt.savefig(f'./figure_plot/dist_plot/mousavi_dist_scatter.png', format='png', transparent=True, dpi=600)
# plt.show()
#
# p_travel_r_squared = r2_score(label_p_travel, pre_p_travel_result)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_p_travel, pre_p_travel_result, color='#FF6347', alpha=0.5)
# plt.plot([0, 30], [0, 30], color='white', linestyle='--', linewidth=3)
# plt.text(0, 20, f'R² = {p_travel_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
# plt.xlabel('True p travel time (s)', fontsize=15)
# plt.ylabel('Predicted p travel time (s)', fontsize=15)
# # plt.legend()
# plt.grid()
# current_xlim = plt.xlim()
# plt.xlim(current_xlim[0], 25)
# current_ylim = plt.ylim()
# plt.ylim(current_ylim[0], 25)
# # plt.savefig(f'./figure_plot/dist_plot/mousavi_time_scatter.png', format='png', transparent=True, dpi=600)
# plt.show()

# ---------------------------------------------------------------------------------------------------------
# Plot the distribution of the estimation back_azimuth
# pre_angles_rad = np.arctan2(pre_azimuth[:, 1], pre_azimuth[:, 0])
# pre_azimuth_deg = np.degrees(pre_angles_rad)
#
# label_angles_rad = np.arctan2(label_azi[:, 1], label_azi[:, 0])
# label_azi_deg = np.degrees(label_angles_rad)
#
# plot_azi_deg = pre_azimuth_deg - label_azi_deg
#
# plot_azi_deg[plot_azi_deg > 180] -= 360
# plot_azi_deg[plot_azi_deg < -180] += 360
#
# mean_error_azi = np.mean(plot_azi_deg)
# mae_azi = np.mean(np.abs(plot_azi_deg))
# sigma_azi = np.std(plot_azi_deg)
#
# bin_counts1, bin_edges1 = np.histogram(plot_azi_deg, bins=60)
# plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
#         color="#4169E1", label='MAE=%.4f \n Std=%.4f' % (mae_azi, sigma_azi))
# plt.legend(fontsize=10)
# plt.xlim(-100, 100)
# plt.xlabel('Azimuth residuals (deg)')
# plt.ylabel('Frequency')
# # plt.savefig(f'./figure_plot/azi_plot/our_azi_model_residual.jpg', format='jpg', transparent=True, dpi=600)
# plt.show()
#
# # dist_r_squared = r2_score(label_azi_deg, pre_azimuth_deg)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_azi_deg, pre_azimuth_deg, color='#4169E1', alpha=0.5)
# plt.plot([0, 360], [0, 360], color='white', linestyle='--', linewidth=3)
# plt.plot([0, 180], [180, 360], color='red', linestyle='--', linewidth=3)
# plt.plot([180, 360], [0, 180], color='red', linestyle='--', linewidth=3)
# # plt.text(100, 300, f'R² = {dist_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
# plt.xlabel('True Azimuth (deg)', fontsize=15)
# plt.ylabel('Predicted Azimuth (deg)', fontsize=15)
# # plt.legend()
# plt.grid()
# # plt.savefig(f'./figure_plot/azi_plot/our_azi_model_scatter.jpg', format='jpg', transparent=True, dpi=600)
# plt.show()

# ---------------------------------------------------------------------------------------------------------
# Calculate the Recall, Precision and F1 score
# number = dt.shape[0]
# num = 0
# d_re = []
# d_pr = []
# d_f1 = []
# for i in range(number):
#     real_d = dt[i, :]
#     pre_d = pre_d_result[i, 0, :]
#     re, pr, f1 = Event_F1(real_d, pre_d)
#     d_re.append(re)
#     d_pr.append(pr)
#     d_f1.append(f1)
#     num += 1
# mean_d_re = np.sum(d_re) / num
# mean_d_pr = np.sum(d_pr) / num
# mean_d_f1 = np.sum(d_f1) / num
#
# mean_p_re, mean_p_pr, mean_p_f1 = recall(pt, pre_p_result)
# mean_s_re, mean_s_pr, mean_s_f1 = recall(st, pre_s_result)
#
# print("event detect: recall: %.4f precision: %.4f f1: %.4f" % (mean_d_re, mean_d_pr, mean_d_f1))
# print("p_picking: recall: %.4f precision: %.4f f1: %.4f" % (mean_p_re, mean_p_pr, mean_p_f1))
# print("s_picking: recall: %.4f precision: %.4f f1: %.4f" % (mean_s_re, mean_s_pr, mean_s_f1))

# ---------------------------------------------------------------------------------------------------------
# You can inspect the results of the network by showing the figure here!
x_data = np.linspace(0, 60, 6000)
for j in range(0, 128):
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.1)

    axs[0].plot(x_data, batch_real_data[0][j, 0, :], 'black', label='E component')
    axs[0].set_xticklabels([])
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[1].plot(x_data, batch_real_data[0][j, 1, :], 'black', label='N component')
    axs[1].set_xticklabels([])
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[2].plot(x_data, batch_real_data[0][j, 2, :], 'black', label=' component')
    axs[2].set_xticklabels([])
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()

    # axs[3].plot(x_data, pre_x_result[j, 0, :], 'black', label='Denoised wave E component')
    # axs[3].set_xticklabels([])
    # axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[3].set_ylabel('Amplitude')
    # axs[3].legend()
    # axs[4].plot(x_data, pre_x_result[j, 1, :], 'black', label='Denoised wave N component')
    # axs[4].set_xticklabels([])
    # axs[4].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[4].set_ylabel('Amplitude')
    # axs[4].legend()
    # axs[5].plot(x_data, pre_x_result[j, 2, :], 'black', label='Denoised wave Z component')
    # axs[5].set_xticklabels([])
    # axs[5].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[5].set_ylabel('Amplitude')
    # axs[5].legend()

    # axs[6].plot(x_data, dt[j, :], 'green', label='Predict_detector_event', linestyle='--')
    # axs[6].plot(x_data, pt[j, :], 'blue', label='Predict_pick_P_wave', linestyle='--')
    # axs[6].plot(x_data, st[j, :], 'yellow', label='Predict_pick_S_wave', linestyle='--')
    # axs[6].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[6].set_ylabel('Probability')
    # axs[6].legend()

    axs[3].plot(x_data, pre_d_result[j, 0, :], '#65ab7c', linestyle='--', linewidth=2, label='Predicted event')
    axs[3].plot(x_data, pre_p_result[j, 0, :], '#665fd1', linestyle='--', linewidth=2, label='Predicted p')
    axs[3].plot(x_data, pre_s_result[j, 0, :], '#c44240', linestyle='--', linewidth=2, label='Predicted s')
    axs[3].set_ylabel('Probability')
    axs[3].set_ylim(0, 1.1)
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')

    labels = ['(I)', '(II)', '(III)']
    for ax, label in zip(axs.flat, labels):
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1))

    plt.show()

    inp = input("Press a key to plot the next waveform!")
    if inp == "r":
        continue

