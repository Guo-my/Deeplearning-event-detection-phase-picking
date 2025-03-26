import numpy as np
import h5py
import os
import torch
import torch.nn as nn
from detect_data_augmentation import DataGenerator
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
np.random.seed(3407)


def read(input_hdf5=None,
         input_csv=None,
         output_name=None,
         input_dimention=(6000, 3),
         batch_size=64,
         shuffle=False,
         label_type='triangle',
         normalization_mode='std',
         augmentation=False,
         add_event_r=0,
         shift_event_r=0.99,
         add_noise_r=0.3,
         drop_channel_r=0.5,
         add_gap_r=0.3,
         coda_ratio=1.4,
         scale_amplitude_r=0.3,
         pre_emphasis=False,
         loss_weights=[0.05, 0.40, 0.55],
         train_valid_test_split=[0.70, 0.15, 0.15],
         mode=None):
    args = {
        "input_hdf5": input_hdf5,
        "input_csv": input_csv,
        "output_name": output_name,
        "input_dimention": input_dimention,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "label_type": label_type,
        "normalization_mode": normalization_mode,
        "augmentation": augmentation,
        "add_event_r": add_event_r,
        "shift_event_r": shift_event_r,
        "add_noise_r": add_noise_r,
        "add_gap_r": add_gap_r,
        "coda_ratio": coda_ratio,
        "drop_channel_r": drop_channel_r,
        "scale_amplitude_r": scale_amplitude_r,
        "pre_emphasis": pre_emphasis,
        "loss_weights": loss_weights,
        "train_valid_test_split": train_valid_test_split,
        }

    params_training = {'file_name': str(args['input_hdf5']),
                       'dim': args['input_dimention'][0],
                       'batch_size': args['batch_size'],
                       'n_channels': args['input_dimention'][-1],
                       'shuffle': args['shuffle'],
                       'norm_mode': args['normalization_mode'],
                       'label_type': args['label_type'],
                       'augmentation': args['augmentation'],
                       'add_event_r': args['add_event_r'],
                       'add_gap_r': args['add_gap_r'],
                       'coda_ratio': args['coda_ratio'],
                       'shift_event_r': args['shift_event_r'],
                       'add_noise_r': args['add_noise_r'],
                       'drop_channe_r': args['drop_channel_r'],
                       'scale_amplitude_r': args['scale_amplitude_r'],
                       'pre_emphasis': args['pre_emphasis']}

    params_validation = {'file_name': str(args['input_hdf5']),
                       'dim': args['input_dimention'][0],
                       'batch_size': args['batch_size'],
                       'n_channels': args['input_dimention'][-1],
                       'shuffle': False,
                       'norm_mode': args['normalization_mode'],
                       'label_type': args['label_type'],
                       'augmentation': False,
                       'add_event_r': args['add_event_r'],
                       'add_gap_r': args['add_gap_r'],
                       'coda_ratio': args['coda_ratio'],
                       'shift_event_r': args['shift_event_r'],
                       'add_noise_r': args['add_noise_r'],
                       'drop_channe_r': args['drop_channel_r'],
                       'scale_amplitude_r': args['scale_amplitude_r'],
                       'pre_emphasis': args['pre_emphasis']}

    params_noise = {'file_name': str(args['input_hdf5']),
                         'dim': args['input_dimention'][0],
                         'batch_size': args['batch_size'],
                         'n_channels': args['input_dimention'][-1],
                         'shuffle': False,
                         'norm_mode': 'max',
                         'augmentation': True}

    training_data, validation_data, test_data = _split_data(args, args["output_name"])
    # training_noise, validation_noise, test_noise = _split_noise(args, args["output_name"])
    # real_data_test = _split_real_data(args, args["output_name"])

    # detect & pick training
    # training = training_data + training_noise
    # validation = validation_data + validation_noise
    # test = test_data + test_noise
    # print('noise', len(training_noise))
    # np.random.shuffle(training)
    # np.random.shuffle(validation)
    # np.random.shuffle(test)
    # np.save(args["output_name"] + '/total_test', test_data)
    training_generator = DataGenerator(training_data, **params_training)
    validation_generator = DataGenerator(validation_data, **params_validation)
    test_generator = DataGenerator(test_data, **params_training)

    # training_generator = DataGenerator(training_data, **params_training)
    # validation_generator = DataGenerator(validation_data, **params_validation)
    # noise_generator = DataGenerator(training_noise, **params_noise)
    # test_generator = DataGenerator(test_data, **params_validation)
    # real_data_test_generator = DataGenerator(real_data_test, **params_validation)

    # if mode == 'train':
    #     training = _split(args, args["output_name"])
    #     generator = DataGenerator(training, **params_training)
    # else:
    #     validation = _split_val(args, args["output_name"])
    #     generator = DataGenerator(validation, **params_validation)
    # if mode == 'test':
    #     return test_generator, real_data_test_generator
    # else:
    #     return training_generator, validation_generator, noise_generator
    if mode == 'test':
        return test_generator
    else:
        return training_generator, validation_generator


def _split_data(args, save_dir):
    df = pd.read_csv(args['input_csv'], low_memory=False)
    df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km.astype(float) < 110)
            & (df.snr_db.str.strip("[]").str.split().str[0].astype(float) > 20)
            & (df.snr_db.str.strip("[]").str.split().str[1].astype(float) > 20)
            & (df.snr_db.str.strip("[]").str.split().str[2].astype(float) > 20)
            & (df.source_depth_km.notna())]
    # df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km.astype(float) < 300)
    #         & (df.snr_db.str.strip("[]").str.split().str[0].astype(float) > 25)
    #         & (df.snr_db.str.strip("[]").str.split().str[1].astype(float) > 25)
    #         & (df.snr_db.str.strip("[]").str.split().str[2].astype(float) > 25)]
    ev_list = df.trace_name.tolist()
    np.random.shuffle(ev_list)
    training = ev_list[:int(args['train_valid_test_split'][0] * len(ev_list))]
    # np.random.shuffle(training)
    validation = ev_list[int(args['train_valid_test_split'][0] * len(ev_list)):
                         int(args['train_valid_test_split'][0] * len(ev_list) + args['train_valid_test_split'][1] * len(
                             ev_list))]
    test = ev_list[
           int(args['train_valid_test_split'][0] * len(ev_list) + args['train_valid_test_split'][1] * len(ev_list)):]
    # np.save(save_dir + '/test', test)
    return training, validation, test


def _split_noise(args, save_dir):
    df = pd.read_csv(args['input_csv'], low_memory=False)
    df = df[(df.trace_category == 'noise')]
    ev_list = df.trace_name.tolist()
    np.random.shuffle(ev_list)
    training = ev_list[:int(args['train_valid_test_split'][0] * len(ev_list))]
    # np.random.shuffle(training)
    validation = ev_list[int(args['train_valid_test_split'][0] * len(ev_list)):
                         int(args['train_valid_test_split'][0] * len(ev_list) + args['train_valid_test_split'][1] * len(
                             ev_list))]
    test = ev_list[
           int(args['train_valid_test_split'][0] * len(ev_list) + args['train_valid_test_split'][1] * len(ev_list)):]
    # np.save(save_dir + '/test', test)
    return training, validation, test


def _split_real_data(args, save_dir):
    df = pd.read_csv(args['input_csv'], low_memory=False)
    df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km.astype(float) < 300)
            & (df.snr_db.str.strip("[]").str.split().str[0].astype(float) < 20)]
    ev_list = df.trace_name.tolist()
    np.random.shuffle(ev_list)
    test = ev_list[:10000]
    return test
