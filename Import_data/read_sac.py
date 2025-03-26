import os
import numpy as np
import obspy
import time
from scipy.signal import decimate, butter, lfilter
import struct
import matplotlib.pyplot as plt
import signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(3):
            y[i, j, :] = lfilter(b, a, data[i, j, :])
    return y


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


def resample(original_signal, original_sampling_rate, target_sampling_rate):
    downsampled_signal = []
    downsample_factor = original_sampling_rate // target_sampling_rate
    for i in range(original_signal.shape[0]):
        cluster = []
        for j in range(3):
            cluster.append(decimate(original_signal[i, j, :], downsample_factor))
        cluster = np.array(cluster)
        downsampled_signal.append(cluster)
    downsampled_signal = np.array(downsampled_signal)
    return downsampled_signal


def padding(signal0, signal_shape, desired_length):
    padded_signal = np.zeros((signal0.shape[0], signal0.shape[1], desired_length))
    padded_signal[:, :, 2999:3500] = signal0[:, :, :]
    return padded_signal


# 定义好解码
class newsac():
    def __init__(self, filename):
        f = open(filename, 'rb')
        hdrBin = f.read(632)
        sfmt = 'f' * 70 + 'I ' * 40 + '8s ' * 22 + '16s';
        hdrFmt = struct.Struct(sfmt)
        self.m_header = hdrFmt.unpack(hdrBin)
        self.npts = int(self.m_header[79])
        self.a = float(self.m_header[8])
        self.b = float(self.m_header[5])
        self.delta = float(self.m_header[0])
        self.t1 = float(self.m_header[11])
        self.t0 = float(self.m_header[10])
        self.t2 = float(self.m_header[12])
        self.t3 = float(self.m_header[13])
        self.t4 = float(self.m_header[14])
        fmt_data = 'f' * self.npts
        dataFmt = struct.Struct(fmt_data)
        dataBin = f.read(4 * self.npts)
        f.close()
        self.m_data = np.array(dataFmt.unpack(dataBin))


class sta():
    def __init__(self, ID='', tp=0, ts=0, E=[], N=[], Z=[]):
        self.ID = ID
        self.tp = tp
        self.ts = ts
        self.E = E
        self.N = N
        self.Z = Z


def custom_sort_key(filename):
    # 提取第三个和第四个字符
    third_char = filename[2] if len(filename) > 2 else ''
    fourth_char = filename[3] if len(filename) > 3 else ''

    # 尝试将第三个和第四个字符转换为数字，如果转换失败则返回字符本身
    third_char_num = ('0', int(third_char)) if third_char.isdigit() else ('1', third_char)
    fourth_char_num = int(fourth_char) if fourth_char.isdigit() else fourth_char

    # 返回排序键：先按照第三个字符的数字、字母排序，再按照第四个字符的数字排序
    return (isinstance(third_char_num, int), third_char_num, isinstance(fourth_char_num, int), fourth_char_num)


print('read all events')
start_time = time.time()
events = []
E = []
N = []
Z = []

filename = os.listdir("./broadband")
filename.remove('broadband.npy')
filename = sorted(filename, key=custom_sort_key)
print(filename)

# 读取相应的数据
for each0 in filename:
    print(each0)
    if each0.endswith('.BHE'):
        E0 = newsac("./broadband/%s" % each0)
        E.append(E0)
    if each0.endswith('.BHN'):
        # print(each0)
        N0 = newsac("./broadband/%s" % each0)
        N.append(N0)
    if each0.endswith('.BHZ'):
        # print(each0)
        Z0 = newsac("./broadband/%s" % each0)
        Z.append(Z0)
for i in range(len(E)):
    E1 = E[i]
    N1 = N[i]
    Z1 = Z[i]
    tp = int((E1.t0 - E1.b) / E1.delta)
    ts = int((E1.t4 - E1.b) / E1.delta)
    tmp = sta(tp=tp, ts=ts, E=E1.m_data, N=N1.m_data, Z=Z1.m_data)
    events.append(tmp)
end_time = time.time()
run_time = end_time - start_time
print('read over')
print('time of read events data:%.2f' % run_time)

# 将数据分别保存至
stream = obspy.Stream()
tensor_list = []
window_length = 10001
# overlap = 0.5
# step_size = int(window_length * (1 - overlap))
window_data = np.zeros((len(events), 3, window_length))
for i, microseismicdata in enumerate(events):
    e_data = microseismicdata.E
    traceE = obspy.Trace(data=e_data)
    # n_samples = len(traceE)
    # print(n_samples)
    # n_windows = (n_samples - window_length) // step_size + 1
    # for i in range(n_windows):
    #     start = i * step_size
    #     end = start + window_length
    window_data[i, 0, :] = traceE.data[:]
    # 归一化
    # window_data = (window_data - np.mean(window_data, axis=0)) / np.std(window_data, axis=0)

    n_data = microseismicdata.N
    traceN = obspy.Trace(data=n_data)
    #         trace_re = trace.resample(sampling_rate=100, strict_length=True, window='hann', no_filter=True)
    # n_samples = len(traceN)
    # n_windows = (n_samples - window_length) // step_size + 1
    # for j in range(n_windows):
    #     start = j * step_size
    #     end = start + window_length
    window_data[i, 1, :] = traceN.data[:]
    # 归一化
    # window_data = (window_data - np.mean(window_data, axis=0)) / np.std(window_data, axis=0)

    z_data = microseismicdata.Z
    traceZ = obspy.Trace(data=z_data)
    #         trace_re = trace.resample(sampling_rate=100, strict_length=True, window='hann', no_filter=True)
    # n_samples = len(traceZ)
    # n_windows = (n_samples - window_length) // step_size + 1
    # for k in range(n_windows):
    #     start = k * step_size
    #     end = start + window_length
    window_data[i, 2, :] = traceZ.data[:]
    # 归一化
    # window_data = (window_data - np.mean(window_data, axis=0)) / np.std(window_data, axis=0)

print(window_data.shape)
print('------------------------------------------------------------')
resampe_signal = resample(window_data, 2000, 100)
print(resampe_signal.shape)
pad_signal = padding(resampe_signal, resampe_signal.shape, 6000)
print(pad_signal.shape)
band_signal = butter_bandpass_filter(pad_signal, 1, 45, 100)
nor_signal = normalize(band_signal)
print(nor_signal.shape)

# x_axis = np.linspace(0, nor_signal.shape[2], nor_signal.shape[2])
# for k in range(nor_signal.shape[0]):
#     fig, axs = plt.subplots(3, 1, figsize=(10, 6))
#     fig.subplots_adjust(hspace=0.3)
#     axs[0].plot(nor_signal[k, 0, :], 'k', label='Event_signal')
#     axs[0].set_xticklabels([])
#     axs[0].set_ylabel('Amplitude')
#     axs[0].legend()
#
#     axs[1].plot(nor_signal[k, 1, :], 'k', label='Event_signal')
#     axs[1].set_xticklabels([])
#     axs[1].set_ylabel('Amplitude')
#     axs[1].legend()
#
#     axs[2].plot(nor_signal[k, 2, :], 'k', label='Event_signal')
#     axs[2].set_ylabel('Amplitude')
#     axs[2].legend()
#     plt.show()
#     inp = input("Press a key to plot the next waveform!")
#     if inp == "r":
#         continue

output_directory = "./broadband"
output_file = "broadband.npy"
np.save(os.path.join(output_directory, output_file), nor_signal)
print('The file is saved in npy format successfully')
