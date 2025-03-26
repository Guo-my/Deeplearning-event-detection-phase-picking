import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import re

file_name = "./merged/merge.hdf5"
csv_file = "./merged/merge.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file, low_memory=False)
print(f'total events in csv file: {len(df)}')


# filterering the dataframe
df = df[(df.trace_category == 'noise')]

wave_ptime = []
wave_stime = []
waves_end = []
print(f'total events selected: {len(df)}')
# df.to_csv('snr_20_data.csv', index=False)
# making a list of trace names for the selected data
ev_list = df['trace_name'].to_list()

# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, 'r')
datalist = []
# for c, evi in enumerate(ev_list):
#     dataset = dtfl.get('data/'+str(evi))
#     # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
#     data = np.array(dataset)
#     datalist.append(data)
#
#     wave_ptime.append(dataset.attrs['p_arrival_sample'])
#     wave_stime.append(dataset.attrs['s_arrival_sample'])
#     waves_end.append(dataset.attrs['coda_end_sample'])

    #
    # for at in dataset.attrs:
    #     print(at, dataset.attrs[at])
    #
    # inp = input("Press a key to plot the next waveform!")
    # if inp == "r":
    #     continue

# wave_ptime = np.array(wave_ptime)
# wave_stime = np.array(wave_stime)
# waves_end = np.array(waves_end).squeeze()
# print(waves_end.shape, wave_ptime.shape)
# np.save('wave_ptime.npy', wave_ptime)
# np.save('wave_stime.npy', wave_stime)
# np.save('waves_end.npy', waves_end)
# traindata = np.array(datalist)
# print(traindata.shape)
# test1 = traindata[0, :, 0]
# test2 = traindata[1, :, 0]
# test3 = traindata[2, :, 0]
# test4 = traindata[3, :, 0]
# output_directory = r"D:\Pycharm\Denoise\DATASETS\DeepDenoiser"
# output_file1 = r"real_data1.npz"
# output_file2 = r"real_data2.npz"
# output_file3 = r"real_data3.npz"
# output_file4 = r"real_data4.npz"
# np.savez(os.path.join(output_directory, output_file1),
#          test1_data=test1)
# np.savez(os.path.join(output_directory, output_file2),
#          test1_data=test2)
# np.savez(os.path.join(output_directory, output_file3),
#          test1_data=test3)
# np.savez(os.path.join(output_directory, output_file4),
#          test1_data=test4)

# print(traindata.shape)
# output_directory = r"D:\Pycharm\Denoise\DATASETS"
# output_file = r"paper_real.npy"
# np.save(os.path.join(output_directory, output_file), traindata)
