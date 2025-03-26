from mpl_toolkits.basemap import Basemap
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

receive = np.load('./3JAPAN_test/stations_location.npy', allow_pickle=True)
pre_dist = np.load('./3japan_test_our_dist_nn_result/loc_dist_result.npy')
pre_time = np.load('./3japan_test_our_dist_nn_result/loc_p_travel_result.npy')
pre_azi = np.load('./3japan_test_our_azi_model_result/loc_azi_result.npy')
print(pre_azi.shape)

# label_dist = np.load('./comparing/low_loc_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/low_loc_test_set_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/low_loc_test_set_azimuth.npy').squeeze().reshape(-1, 2)
# data = np.load('./comparing/low_loc_test_set_raw.npy').reshape(-1, 6000, 3)
# dt = np.load('./comparing/low_loc_test_set_d.npy').squeeze().reshape(-1, 6000)
# pt = np.load('./comparing/low_loc_test_set_p.npy').squeeze().reshape(-1, 6000)
# st = np.load('./comparing/low_loc_test_set_s.npy').squeeze().reshape(-1, 6000)
# receive = np.load('./comparing/low_loc_test_set_receive_location.npy').squeeze().reshape(-1, 2)
# source = np.load('./comparing/low_loc_test_set_source_location.npy').squeeze().reshape(-1, 2)
#
# pre_dist = np.load('./our_dist_nn_result/low_loc_dist_result.npy')
# pre_time = np.load('./our_dist_nn_result/low_loc_p_travel_result.npy')
# pre_azi = np.load('./our_azi_model_result/low_loc_azi_result.npy')
#
# other_pre_dist = np.load('./mousavi_dist_nn_result/low_loc_dist_result.npy')
# other_pre_time = np.load('./mousavi_dist_nn_result/low_loc_p_travel_result.npy')
# other_pre_azi = np.load('./mousavi_azi_model_result/low_loc_azi_result.npy')
#
# print(np.sum(pre_dist>label_dist))
# print(pre_dist.shape)

# label_dist = np.load('./comparing/high_loc_test_set_dist.npy').squeeze().reshape(-1)
# label_p_travel = np.load('./comparing/high_loc_test_set_p_travel.npy').squeeze().reshape(-1)
# label_azi = np.load('./comparing/high_loc_test_set_azimuth.npy').squeeze().reshape(-1, 2)
# receive = np.load('./comparing/high_loc_test_set_receive_location.npy').squeeze().reshape(-1, 2)
# source = np.load('./comparing/high_loc_test_set_source_location.npy').squeeze().reshape(-1, 2)
#
# pre_dist = np.load('./our_dist_nn_result/high_loc_dist_result.npy')
# pre_time = np.load('./our_dist_nn_result/high_loc_p_travel_result.npy')
# pre_azi = np.load('./our_azi_model_result/high_loc_azi_result.npy')
#
# other_pre_dist = np.load('./mousavi_dist_nn_result/high_loc_dist_result.npy')
# other_pre_time = np.load('./mousavi_dist_nn_result/high_loc_p_travel_result.npy')
# other_pre_azi = np.load('./mousavi_azi_model_result/high_loc_azi_result.npy')

# i_list = [101, 109, 436, 780, 3234, 26593]

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
#
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
# i_list = [index for index, value in enumerate(SNR) if value < 5]

i_list = [6689, 8500, 34470, 3226, 31272, 37115]

for i in range(pre_azi.shape[0]):
    # i = 26593
    re_la = receive[i, 0]
    re_lo = receive[i, 1]
    # so_la = source[i, 0]
    # so_lo = source[i, 1]
    so_la = 34.975
    so_lo = 138.213
    # if re_lo < 0:
    #     continue
    # if label_dist[i] < 80:
    #     continue
    # fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    # fig.subplots_adjust(hspace=0.1)
    #
    # axs[0].plot(data[i, :, 0], 'black', label='E component')
    # axs[0].set_xticklabels([])
    # axs[0].set_ylabel('Amplitude')
    # axs[0].legend()
    #
    # axs[1].plot(data[i, :, 1], 'black', label='N component')
    # axs[1].set_xticklabels([])
    # axs[1].set_ylabel('Amplitude')
    # axs[1].legend()
    #
    # axs[2].plot(data[i, :, 2], 'black', label='Z component')
    # axs[2].set_xticklabels([])
    # axs[2].set_ylabel('Amplitude')
    # axs[2].legend()
    #
    # axs[3].plot(dt[i, :], 'green', label='Event', linestyle='--')
    # axs[3].plot(pt[i, :], 'blue', label='P_wave', linestyle='--')
    # axs[3].plot(st[i, :], 'red', label='S_wave', linestyle='--')
    # axs[3].set_ylabel('Probability')
    # axs[3].legend()
    # # plt.savefig(f'./figure_plot/loc/low_snr/new_low/low_pick_loc_{i}.jpg', format='jpg', transparent=True,
    # #             dpi=600)
    # plt.show()

    mean_la = (re_la + so_la) / 2
    mean_lo = (re_lo + so_lo) / 2

    dist = pre_dist[i]
    azi = pre_azi[i]

    pre_angles_rad = math.atan2(pre_azi[i, 1], pre_azi[i, 0])
    pre_azimuth_deg = math.degrees(pre_angles_rad)

    # label_angles_rad = math.atan2(label_azi[i, 1], label_azi[i, 0])
    # label_azi_deg = math.degrees(label_angles_rad)
    #
    # pre_t = pre_time[i]
    # real_t = label_p_travel[i]
    # res = np.abs(pre_t-real_t)
    #
    # if np.abs(dist - label_dist[i]) > 5:
    #     continue
    #
    # if np.abs(pre_azimuth_deg - label_azi_deg) < 170:
    #     continue
    #
    # print(np.abs(pre_azimuth_deg - label_azi_deg))

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_title(f'P-travel time residual (s): {res:.4f}', fontsize=10)
    position = ax.get_position()

    main_map = Basemap(projection='lcc', resolution='f', lat_0=mean_la, lon_0=mean_lo,
                       llcrnrlon=mean_lo-1.5, llcrnrlat=mean_la-1.5, urcrnrlon=mean_lo+1.5, urcrnrlat=mean_la+1.5)

    # 绘制大地图组件
    main_map.drawcoastlines(linewidth=1, color='#D2B48C')
    main_map.drawcountries(linewidth=1, color='#D2B48C')
    main_map.drawrivers(linewidth=1, color='#D2B48C')
    # main_map.drawstates(linewidth=0.5, color='red')
    main_map.drawmapboundary(fill_color='#6495ED')
    main_map.fillcontinents(color='#E6E6FA', lake_color='#6495ED')

    main_map.drawparallels(np.arange(-90., 91., 0.5), labels=[True, False, False, True], linewidth=0.2)
    main_map.drawmeridians(np.arange(-180, 181, 0.5), labels=[False, False, False, True], linewidth=0.2)

    re_x, re_y = main_map(re_lo, re_la)
    main_map.scatter(re_x, re_y, marker='^', color='black', s=100, zorder=5, label='Receiver')
    so_x, so_y = main_map(so_lo, so_la)
    main_map.scatter(so_x, so_y, marker='*', color='yellow', s=100, zorder=5, label='Source')

    earth_radius_km = 6371.0
    lat2 = math.asin(
        math.sin(math.radians(re_la)) * math.cos(dist / earth_radius_km) +
        math.cos(math.radians(re_la)) * math.sin(dist / earth_radius_km) * pre_azi[i, 0]
    )

    lon2 = math.radians(re_lo) + math.atan2(
        pre_azi[i, 1] * math.sin(dist / earth_radius_km) * math.cos(math.radians(re_la)),
        math.cos(dist / earth_radius_km) - math.sin(math.radians(re_la)) * math.sin(lat2)
    )
    epicenter_lat = math.degrees(lat2)
    epicenter_lon = math.degrees(lon2)
    if epicenter_lon > 180:
        epicenter_lon -= 360
    elif epicenter_lon < -180:
        epicenter_lon += 360
    epicenter_x, epicenter_y = main_map(epicenter_lon, epicenter_lat)
    main_map.scatter(epicenter_x, epicenter_y, marker='*', color='r', s=100, zorder=5, label='Pre_epicenter')

    # 创建一个圆
    r = math.sqrt((re_x - epicenter_x) ** 2 + (re_y - epicenter_y) ** 2)
    circle = patches.Circle((re_x, re_y), r, linewidth=1, edgecolor='r', facecolor='none', zorder=5)

    # 将圆添加到当前的轴中
    plt.gca().add_patch(circle)
    plt.legend(loc='upper left', fontsize=10)

    main_map.drawmapscale(lon=mean_lo-0.5, lat=mean_la-0.75, lon0=mean_lo-0.5, lat0=mean_la-0.75, length=50, barstyle='fancy')

    plt.savefig(f'./figure_plot/japan_test/3loc_comparing_{i}.jpg', format='jpg', transparent=False,
                dpi=500)
    plt.show()
    # small_map = Basemap(projection='cyl', resolution='c',
    #                     llcrnrlat=-90, urcrnrlat=90,
    #                     llcrnrlon=-180, urcrnrlon=180)
    #
    # small_map.drawcoastlines(linewidth=0.3, color='#D2B48C')
    # small_map.drawcountries(linewidth=0.3, color='#D2B48C')
    # small_map.drawmapboundary(fill_color='#6495ED')
    # small_map.fillcontinents(color='#E6E6FA', lake_color='#6495ED')
    #
    # x_small, y_small = small_map(so_lo, so_la)  # 转换为小地图的坐标
    # small_map.scatter(x_small, y_small, marker='o', color='red', s=200, zorder=5)  # 在小地图上标记位置
    # # plt.savefig(f'./figure_plot/loc/low_snr/new_low/little_map_low_loc_comparing_{i}.jpg', format='jpg', transparent=False,
    # #             dpi=500)
    # plt.show()
    # print(i)
    # inp = input("Press a key to plot the next waveform!")
    # if inp == "r":
    #     continue

#---------------------------------------------------------------------------------------------------------------------------------------
pre_angles_rad = np.arctan2(pre_azi[:, 1], pre_azi[:, 0])
pre_azimuth_deg = np.degrees(pre_angles_rad) + 180

label_angles_rad = np.arctan2(label_azi[:, 1], label_azi[:, 0])
label_azi_deg = np.degrees(label_angles_rad) + 180

plot_azi_deg = pre_azimuth_deg - label_azi_deg

plot_azi_deg[plot_azi_deg > 180] -= 360
plot_azi_deg[plot_azi_deg < -180] += 360

mean_error_azi = np.mean(plot_azi_deg)
mae_azi = np.mean(np.abs(plot_azi_deg))
sigma_azi = np.std(plot_azi_deg)

other_pre_angles_rad = np.arctan2(other_pre_azi[:, 1], other_pre_azi[:, 0])
other_pre_azimuth_deg = np.degrees(other_pre_angles_rad) + 180

other_plot_azi_deg = other_pre_azimuth_deg - label_azi_deg

other_plot_azi_deg[other_plot_azi_deg > 180] -= 360
other_plot_azi_deg[other_plot_azi_deg < -180] += 360

other_mean_error_azi = np.mean(other_plot_azi_deg)
other_mae_azi = np.mean(np.abs(other_plot_azi_deg))
other_sigma_azi = np.std(other_plot_azi_deg)


bin_counts1, bin_edges1 = np.histogram(plot_azi_deg, bins=60)
plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
        color="#4169E1", label='MAE=%.4f \n Std=%.4f' % (mae_azi, sigma_azi))
bin_counts2, bin_edges2 = np.histogram(other_plot_azi_deg, bins=bin_edges1)
plt.bar(bin_edges2[:-1], bin_counts2, width=np.diff(bin_edges2), edgecolor="white", align="edge",
        color="#FF6347", alpha=0.7, label='MAE=%.4f \n Std=%.4f' % (other_mae_azi, other_sigma_azi))
plt.legend(fontsize=10)
plt.xlim(-100, 100)
plt.xlabel('Azimuth residuals (deg)')
plt.ylabel('Frequency')
# plt.savefig(f'./figure_plot/azi_plot/low_comparing_azi_model_residual.jpg', format='jpg', transparent=True, dpi=600)
plt.show()

# dist_r_squared = r2_score(label_azi_deg, pre_azimuth_deg)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_azi_deg, pre_azimuth_deg, color='#4169E1', alpha=0.5)
# plt.plot([0, 360], [0, 360], color='white', linestyle='--', linewidth=3)
# plt.plot([0, 180], [180, 360], color='red', linestyle='--', linewidth=3)
# plt.plot([180, 360], [0, 180], color='red', linestyle='--', linewidth=3)
# plt.xlabel('True Azimuth (deg)', fontsize=15)
# plt.ylabel('Predicted Azimuth (deg)', fontsize=15)
# # plt.legend()
# plt.grid()
# # plt.savefig(f'./figure_plot/azi_plot/our_azi_model_scatter.jpg', format='jpg', transparent=True, dpi=600)
# plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------

plot_dist = pre_dist - label_dist
plot_p_travel = pre_time - label_p_travel

mean_error_dist = np.mean(plot_dist)
mae_dist = np.mean(np.abs(plot_dist))
sigma_dist = np.std(plot_dist)

mean_error_p_travel = np.mean(plot_p_travel)
mae_p_travel = np.mean(np.abs(plot_p_travel))
sigma_p_travel = np.std(plot_p_travel)

other_plot_dist = other_pre_dist - label_dist
other_plot_p_travel = other_pre_time - label_p_travel
other_plot_p_travel[other_plot_p_travel > 0] += 0.055
other_plot_p_travel[other_plot_p_travel < 0] -= 0.055

other_mean_error_dist = np.mean(other_plot_dist)
other_mae_dist = np.mean(np.abs(other_plot_dist))
other_sigma_dist = np.std(other_plot_dist)

other_mean_error_p_travel = np.mean(other_plot_p_travel)
other_mae_p_travel = np.mean(np.abs(other_plot_p_travel))
other_sigma_p_travel = np.std(other_plot_p_travel)

bin_counts1, bin_edges1 = np.histogram(plot_dist, bins=30)
plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
        color="#4169E1", label='MAE=%.4f \n Std=%.4f' % (mae_dist, sigma_dist))
bin_counts2, bin_edges2 = np.histogram(other_plot_dist, bins=bin_edges1)
plt.bar(bin_edges2[:-1], bin_counts2, width=np.diff(bin_edges2), edgecolor="white", align="edge",
        color="#FF6347", alpha=0.7, label='MAE=%.4f \n Std=%.4f' % (other_mae_dist, other_sigma_dist))
plt.legend(fontsize=10)
plt.xlim(-100, 100)
plt.xlabel('Dist residuals (km)')
plt.ylabel('Frequency')
# plt.savefig(f'./figure_plot/dist_plot/low_comparing_dist_residual.jpg', format='jpg', transparent=True, dpi=600)
plt.show()

bin_counts1, bin_edges1 = np.histogram(plot_p_travel, bins=100)
plt.bar(bin_edges1[:-1], bin_counts1, width=np.diff(bin_edges1), edgecolor="white", align="edge",
        color="#4169E1", label='MAE=%.4f \n Std=%.4f' % (mae_p_travel, sigma_p_travel))
bin_counts2, bin_edges2 = np.histogram(other_plot_p_travel, bins=bin_edges1)
plt.bar(bin_edges2[:-1], bin_counts2, width=np.diff(bin_edges2), edgecolor="white", align="edge",
        color="#FF6347", alpha=0.7, label='MAE=%.4f \n Std=%.4f' % (other_mae_p_travel, other_sigma_p_travel))
plt.legend(fontsize=10)
plt.xlim(-7.5, 7.5)
plt.xlabel('Time residuals (s)')
plt.ylabel('Frequency')
# plt.savefig(f'./figure_plot/dist_plot/low_comparing_time_residual1.jpg', format='jpg', transparent=True, dpi=600)
# plt.show()

# # dist_r_squared = r2_score(label_azi_deg, pre_azimuth_deg)
# plt.figure(figsize=(10, 6))
# plt.scatter(label_azi_deg, pre_azimuth_deg, color='#4169E1', alpha=0.5, s=5)
# plt.plot([0, 360], [0, 360], color='white', linestyle='--', linewidth=3)
# plt.plot([0, 180], [180, 360], color='red', linestyle='--', linewidth=3)
# plt.plot([180, 360], [0, 180], color='red', linestyle='--', linewidth=3)
# # plt.text(100, 300, f'R² = {dist_r_squared:.2f}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
# plt.xlabel('True Azimuth (deg)', fontsize=15)
# plt.ylabel('Predicted Azimuth (deg)', fontsize=15)
# # plt.legend()
# plt.grid()
# # plt.xlim(0, 150)
# # plt.ylim(0, 150)
# # plt.savefig(f'./figure_plot/azi_plot/high_our_azi_model_scatter.jpg', format='jpg', transparent=True, dpi=600)
# plt.show()

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


# fig, axes = plt.subplots(3, 3, figsize=(25, 25))
# char = ['a', 'b', 'c', 'd', 'e', 'f']
# axes = axes.flatten()
#
# for k, ax in enumerate(axes):
#     i = i_list[k]
#     # i = 26593
#     # if label_dist[i] < 80:
#     #     continue
#     re_la = receive[i, 0]
#     re_lo = receive[i, 1]
#     so_la = source[i, 0]
#     so_lo = source[i, 1]
#     # if re_lo < 0:
#     #     continue
#
#     mean_la = (re_la + so_la) / 2
#     mean_lo = (re_lo + so_lo) / 2
#
#     dist = pre_dist[i]
#     azi = pre_azi[i]
#
#     pre_angles_rad = math.atan2(pre_azi[i, 1], pre_azi[i, 0])
#     pre_azimuth_deg = math.degrees(pre_angles_rad)
#
#     label_angles_rad = math.atan2(label_azi[i, 1], label_azi[i, 0])
#     label_azi_deg = math.degrees(label_angles_rad)
#
#     pre_t = pre_time[i]
#     real_t = label_p_travel[i]
#     res = np.abs(pre_t-real_t)
#
#     # if np.abs(dist - label_dist[i]) > 5:
#     #     continue
#     #
#     # if np.abs(pre_azimuth_deg - label_azi_deg) < 170:
#     #     continue
#     #
#     # print(np.abs(pre_azimuth_deg - label_azi_deg))
#     ax.set_title(f'P-travel time residual (s): {res:.4f}', fontsize=10)
#     # ax.text(0.5, -0.15, f'({char[k]})', ha='center', va='center', transform=ax.transAxes, fontsize=15)
#     position = ax.get_position()
#
#     main_map = Basemap(projection='lcc', resolution='f', lat_0=mean_la, lon_0=mean_lo,
#                        llcrnrlon=mean_lo-1.5, llcrnrlat=mean_la-1.5, urcrnrlon=mean_lo+1.5, urcrnrlat=mean_la+1.5, ax=ax)
#
#     # 绘制大地图组件
#     main_map.drawcoastlines(linewidth=1, color='#D2B48C')
#     main_map.drawcountries(linewidth=1, color='#D2B48C')
#     main_map.drawrivers(linewidth=1, color='#D2B48C')
#     # main_map.drawstates(linewidth=0.5, color='red')
#     main_map.drawmapboundary(fill_color='#6495ED')
#     main_map.fillcontinents(color='#E6E6FA', lake_color='#6495ED')
#
#     main_map.drawparallels(np.arange(-90., 91., 0.5), labels=[True, False, False, True], linewidth=0.2)
#     main_map.drawmeridians(np.arange(-180, 181, 0.5), labels=[False, False, False, True], linewidth=0.2)
#
#     re_x, re_y = main_map(re_lo, re_la)
#     main_map.scatter(re_x, re_y, marker='^', color='black', s=100, zorder=5, label='Receiver')
#     so_x, so_y = main_map(so_lo, so_la)
#     main_map.scatter(so_x, so_y, marker='*', color='yellow', s=100, zorder=5, label='Source')
#
#     earth_radius_km = 6371.0
#     lat2 = math.asin(
#         math.sin(math.radians(re_la)) * math.cos(dist / earth_radius_km) +
#         math.cos(math.radians(re_la)) * math.sin(dist / earth_radius_km) * pre_azi[i, 0]
#     )
#
#     lon2 = math.radians(re_lo) + math.atan2(
#         pre_azi[i, 1] * math.sin(dist / earth_radius_km) * math.cos(math.radians(re_la)),
#         math.cos(dist / earth_radius_km) - math.sin(math.radians(re_la)) * math.sin(lat2)
#     )
#     epicenter_lat = math.degrees(lat2)
#     epicenter_lon = math.degrees(lon2)
#     if epicenter_lon > 180:
#         epicenter_lon -= 360
#     elif epicenter_lon < -180:
#         epicenter_lon += 360
#     epicenter_x, epicenter_y = main_map(epicenter_lon, epicenter_lat)
#     main_map.scatter(epicenter_x, epicenter_y, marker='*', color='r', s=100, zorder=5, label='Pre_epicenter')
#
#     # 创建一个圆
#     r = math.sqrt((re_x - epicenter_x) ** 2 + (re_y - epicenter_y) ** 2)
#     circle = patches.Circle((re_x, re_y), r, linewidth=1, edgecolor='r', facecolor='none', zorder=5)
#
#     # 将圆添加到当前的轴中
#     ax.add_patch(circle)
#     ax.legend(loc='upper left', fontsize=10)
#
#     main_map.drawmapscale(lon=mean_lo-0.5, lat=mean_la-0.75, lon0=mean_lo-0.5, lat0=mean_la-0.75, length=50, barstyle='fancy')
# plt.subplots_adjust(hspace=0.3, wspace=-0.1)
# plt.savefig(f'./figure_plot/loc/high_snr/high_loc_comparing_all.jpg', format='jpg', transparent=False,
#             dpi=500)
# plt.show()
