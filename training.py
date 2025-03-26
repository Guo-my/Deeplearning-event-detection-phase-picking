import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
import argparse
import random
import matplotlib.pyplot as plt
from Read_data import read
from multiprocessing import Pool
# from Networkmodle import DenoiseAutoEncoder
from .model.Detect_pick_model import Generalist_without_denoise
random.seed(3407)

t = np.linspace(0, 60, 6000, endpoint=True)

def ssq_function(signals):
    wavelet_coefficients = torch.zeros((signals.shape[0], 3, 257, 47))
    for index, group in enumerate(signals):
        group_coeffs = torch.zeros((3, 257, 47))
        for i in range(3):
            component = group[i, :]
            ssq_values = torch.stft(component, n_fft=512, hop_length=128, window=torch.hann_window(512), return_complex=True)
            magnitude = torch.abs(ssq_values)
            magnitude_db = 10 * torch.log10(magnitude + 1e-10)
            # real = ssq_values.real
            # imag = ssq_values.imag
            group_coeffs[i, :, :] = magnitude_db
        # group_coeffs = torch.tensor(group_coeffs).reshape(6, real.size(0), real.size(1))
        wavelet_coefficients[index, :, :, :] = group_coeffs
    return wavelet_coefficients


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        # new_tensor = 2 * torch.ones_like(targets)
        # new_tensor[targets == 0] = 1
        bce_loss = self.bce_loss(inputs, targets)
        # bce_loss = torch.clamp(bce_loss, min=1e-8)
        # pt = torch.exp(-bce_loss)
        focal_loss = (torch.abs(targets - inputs) ** self.gamma) * bce_loss
        return focal_loss.mean()

class CrossEntropyLossWithProbs(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithProbs, self).__init__()

    def forward(self, prob, target_probs):
        loss = -torch.sum(target_probs * torch.log(prob))

        return loss.mean()


file_name = "../STEAD/merged/merge.hdf5"
csv_file = "../STEAD/merged/merge.csv"
# training_data, validation, noise = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=True, mode='train')
# test, realdata_test = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=False, mode='test')
training_data, validation = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=True, mode='train')
test = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=False, mode='test')


def test_process_batch(i):
    one_batch_clean, one_batch_d, one_batch_p, one_batch_s, one_batch_dist, one_batch_p_travel, one_batch_deep, one_batch_azimuth = test.__getitem__(i)
    return (np.array(one_batch_clean), np.array(one_batch_d),
            np.array(one_batch_p), np.array(one_batch_s), np.array(one_batch_dist),
            np.array(one_batch_p_travel), np.array(one_batch_deep), np.array(one_batch_azimuth))


def val_process_batch(i):
    one_batch_clean, one_batch_d, one_batch_p, one_batch_s, one_batch_dist, one_batch_p_travel, one_batch_deep, one_batch_azimuth = validation.__getitem__(i)
    return (np.array(one_batch_clean), np.array(one_batch_d),
            np.array(one_batch_p), np.array(one_batch_s), np.array(one_batch_dist),
            np.array(one_batch_p_travel), np.array(one_batch_deep), np.array(one_batch_azimuth))


def train_process_batch(i):
    one_batch_clean, one_batch_d, one_batch_p, one_batch_s, one_batch_dist, one_batch_p_travel, one_batch_deep, one_batch_azimuth = training_data.__getitem__(i)
    return (np.array(one_batch_clean), np.array(one_batch_d),
            np.array(one_batch_p), np.array(one_batch_s), np.array(one_batch_dist),
            np.array(one_batch_p_travel), np.array(one_batch_deep), np.array(one_batch_azimuth))


# def noise_process_batch(i):
#     one_batch_noise, _, _, _, _, _, _ = noise.__getitem__(i)
#     return np.array(one_batch_noise)


def cov(data):
    output = []
    for i in range(128):
        one_data = data[i, :, :]
        cov_matrix = np.cov(one_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = eigenvalues[np.newaxis, :]
        cov_mat = np.concatenate((cov_matrix, eigenvalues, eigenvectors), axis=0)
        output.append(cov_mat)
    output = np.array(output)
    return output


def baz(data, p):
    output = []
    for i in range(128):
        x = np.zeros((150, 3))
        one_data = data[i, :, :]
        # plt.plot(p[i, 0, :].cpu())
        # plt.show()
        max_p = np.where(p[i, :, 0] == 1)[0]
        start = max(0, int(max_p) - 50)
        end = min(6001, int(max_p) + 100)
        x[-int(end - start):, :] = one_data[start:end, :]
        output.append(x)
    output = np.array(output)
    return output


def main(opt):
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    OUT_PATH = opt.out_path
    LR = opt.lr
    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available.")
    print("epochs:", NUM_EPOCHS)
    print("batch_size:", BATCH_SIZE)
    print("out_path:", OUT_PATH)
    print("lr:", LR)
    print('Begin to process the validation set')
    print('-----------------------------------------------------------------------------------------------------------')
    pool = Pool()
    val_length = int(len(validation))
    val_results = list(
        tqdm(pool.imap(val_process_batch, range(val_length)), total=val_length))
    validation_c, validation_d, validation_p, validation_s, validation_dist, validation_p_travel, validation_deep, validation_azimuth = [list(t) for t in zip(*val_results)]
    print(validation_c[1].shape)

    print('-----------------------------------------------------------------------------------------------------------')
    # test_length = int(len(test))
    # test_results = list(
    #     tqdm(pool.imap(test_process_batch, range(test_length)), total=test_length))
    # test_c, test_d, test_p, test_s, test_dist, test_p_travel, test_deep, test_azimuth = [list(t) for t in zip(*test_results)]
    # print(test_c[1].shape)
    # np.save('./comparing/test_set_raw', test_c)
    # np.save('./comparing/test_set_d', test_d)
    # np.save('./comparing/test_set_p', test_p)
    # np.save('./comparing/test_set_s', test_s)
    # np.save('./comparing/test_set_dist', test_dist)
    # np.save('./comparing/test_set_p_travel', test_p_travel)
    # np.save('./comparing/test_set_deep', test_deep)
    # np.save('./comparing/test_set_azimuth', test_azimuth)

    print('-----------------------------------------------------------------------------------------------------------')
    # realdata_test_length = int(len(realdata_test))
    # realdata_test_results = list(
    #     tqdm(pool.imap(realdata_test_process_batch, range(realdata_test_length)), total=realdata_test_length))
    # realdata_test_c, realdata_test_d, realdata_test_p, realdata_test_s = [list(t) for t in zip(*realdata_test_results)]
    # print(realdata_test_c[1].shape)

    print('Begin to process the training data')
    print('-----------------------------------------------------------------------------------------------------------')
    train_length = int(len(training_data))
    results = list(
        tqdm(pool.imap(train_process_batch, range(train_length)), total=train_length))
    c, d, p, s, dist, p_travel, deep, azimuth = [list(t) for t in zip(*results)]
    # comb_c = c + c
    # comb_d = d + d
    # comb_p = p + p
    # comb_s = s + s

    indices = list(range(len(c)))
    random.shuffle(indices)
    c_shuffled = [c[i] for i in indices]
    d_shuffled = [d[i] for i in indices]
    p_shuffled = [p[i] for i in indices]
    s_shuffled = [s[i] for i in indices]
    dist_shuffled = [dist[i] for i in indices]
    p_travel_shuffled = [p_travel[i] for i in indices]
    deep_shuffled = [deep[i] for i in indices]
    azimuth_shuffled = [azimuth[i] for i in indices]

    # noise_length = int(len(noise))
    # noise_results = list(
    #     tqdm(pool.imap(noise_process_batch, range(noise_length)), total=noise_length))
    # random.shuffle(noise_results)

    # noise_data = []
    # for i in range(len(c_shuffled)):
    #     data = c_shuffled[i]
    #     pure_noise = noise_results[i]
    #     noisy_data = np.zeros_like(data)
    #     for m in range(data.shape[0]):
    #         noisy_data[m, :, 0] = data[m, :, 0] + pure_noise[m, :, 0] * (np.random.uniform(0.01, 0.45) * np.max(data[m, :, 0]))
    #         noisy_data[m, :, 1] = data[m, :, 1] + pure_noise[m, :, 1] * (np.random.uniform(0.01, 0.45) * np.max(data[m, :, 1]))
    #         noisy_data[m, :, 2] = data[m, :, 2] + pure_noise[m, :, 2] * (np.random.uniform(0.01, 0.45) * np.max(data[m, :, 2]))
    #     noise_data.append(noisy_data)
    #
    # val_noise_data = []
    # for i in range(len(validation_c)):
    #     data = validation_c[i]
    #     pure_noise = noise_results[i+len(c_shuffled)]
    #     noisy_data = np.zeros_like(data)
    #     for m in range(data.shape[0]):
    #         noisy_data[m, :, 0] = data[m, :, 0] + pure_noise[m, :, 0] * (np.random.uniform(0.01, 0.45) * np.max(data[m, :, 0]))
    #         noisy_data[m, :, 1] = data[m, :, 1] + pure_noise[m, :, 1] * (np.random.uniform(0.01, 0.45) * np.max(data[m, :, 1]))
    #         noisy_data[m, :, 2] = data[m, :, 2] + pure_noise[m, :, 2] * (np.random.uniform(0.01, 0.45) * np.max(data[m, :, 2]))
    #     val_noise_data.append(noisy_data)

    # test_noise_data = []
    # for i in range(len(test_c)):
    #     data = test_c[i]
    #     pure_noise = noise_results[i+len(c_shuffled)+len(validation_c)]
    #     noisy_data = np.zeros_like(data)
    #     for m in range(data.shape[0]):
    #         noisy_data[m, :, 0] = data[m, :, 0] + pure_noise[m, :, 0] * (np.random.uniform(0.1, 0.45) * np.max(data[m, :, 0]))
    #         noisy_data[m, :, 1] = data[m, :, 1] + pure_noise[m, :, 1] * (np.random.uniform(0.1, 0.45) * np.max(data[m, :, 1]))
    #         noisy_data[m, :, 2] = data[m, :, 2] + pure_noise[m, :, 2] * (np.random.uniform(0.1, 0.45) * np.max(data[m, :, 2]))
    #     test_noise_data.append(noisy_data)

    print('All the data are processed over!')

    start_time = time.time()

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    argsDict = opt.__dict__
    with open(OUT_PATH + "/hyperparameter.txt", "w") as f:
        f.writelines("-" * 10 + "start" + "-" * 10 + "\n")
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " :    " + str(value) + "\n")
        f.writelines("-" * 10 + "end" + "-" * 10)

    model_path = os.path.join(OUT_PATH, "our_dist_model_withoutps")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    statistics_path = os.path.join(OUT_PATH, "statistics")
    if not os.path.exists(statistics_path):
        os.makedirs(statistics_path)

    loss_path = os.path.join(OUT_PATH, "train_loss")
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    # model = Phase_net()
    model = Generalist_without_denoise()
    # model = dist_nn()
    # model = CR_NN()
    # model = our_bazi_net()
    # model = mousavi_azi_nn()
    # model = our_dist()
    # model = EQtransformer()
    model.weights_init()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {total_params} trainable parameters")

    # initialize
    # optimizer = torch.optim.SGD(DAEmodel.parameters(), lr=LR, momentum=MOMENTUM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-12)
    loss_func = nn.BCELoss()
    loss_mse = nn.MSELoss()
    focal_loss = FocalLoss()
    # qf_loss = QFLoss()
    # phase_net_loss = CrossEntropyLossWithProbs()

    save_metric = 0.0
    train_loss = []
    val_loss = []

    train_pick_loss = []
    train_p_travel_loss = []
    train_dist_loss = []
    train_deep_loss = []
    train_azi_loss = []

    val_pick_loss = []
    val_p_travel_loss = []
    val_dist_loss = []
    val_azi_loss = []
    val_deep_loss = []

    # train_length = int(len(training_data))

    # training
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(range(len(c_shuffled)), colour="white")
        running_results = {'batch_sizes': 0, "loss": 0, "pick_loss": 0, "p_travel_loss": 0, "dist_loss": 0, "deep_loss": 0, "azi_loss": 0}

        model.train()
        for i in train_bar:
            # one_batch_clean, one_batch_x1, one_batch_x2, one_batch_d, one_batch_p, one_batch_s = training_data.__getitem__(i)
            # label_x = torch.tensor(one_batch_clean, dtype=torch.float32).permute(0, 2, 1)
            # train_x1 = torch.tensor(one_batch_x1, dtype=torch.float32).permute(0, 2, 1)
            # train_x2 = torch.tensor(one_batch_x2, dtype=torch.float32).permute(0, 2, 1)
            # label_d = torch.tensor(one_batch_d, dtype=torch.float32).permute(0, 2, 1)
            # label_p = torch.tensor(one_batch_p, dtype=torch.float32).permute(0, 2, 1)
            # label_s = torch.tensor(one_batch_s, dtype=torch.float32).permute(0, 2, 1)
            # train_x = torch.tensor(noise_data[i], dtype=torch.float32).permute(0, 2, 1)

            # baz_x = baz(c_shuffled[i], p_shuffled[i])
            # cov_c = cov(baz_x)
            # baz_x = torch.tensor(baz_x, dtype=torch.float32).permute(0, 2, 1)
            # cov_c = torch.tensor(cov_c, dtype=torch.float32)

            label_x = torch.tensor(c_shuffled[i], dtype=torch.float32).permute(0, 2, 1)
            # label_d = torch.tensor(d_shuffled[i], dtype=torch.float32).permute(0, 2, 1)
            # label_p = torch.tensor(p_shuffled[i], dtype=torch.float32).permute(0, 2, 1)
            # label_s = torch.tensor(s_shuffled[i], dtype=torch.float32).permute(0, 2, 1)
            label_dist = torch.tensor(dist_shuffled[i], dtype=torch.float32)
            label_p_travel = torch.tensor(p_travel_shuffled[i], dtype=torch.float32)
            # label_deep = torch.tensor(deep_shuffled[i], dtype=torch.float32)
            # label_azimuth = torch.tensor(azimuth_shuffled[i], dtype=torch.float32)

            label_x = label_x.to(device)
            # label_d = label_d.to(device)
            # label_p = label_p.to(device)
            # label_s = label_s.to(device)
            label_dist = label_dist.to(device)
            label_p_travel = label_p_travel.to(device)
            # label_deep = label_deep.to(device)
            # label_azimuth = label_azimuth.to(device)
            # train_x_wavelet = stft_function(train_x)
            # train_data = torch.cat((train_x, train_x_wavelet), dim=1)
            batch_size = label_x.size(0)
            running_results["batch_sizes"] += batch_size
            # output_x, output_d, output_p, output_s, output_dist, output_azi = model(train_x)
            output_dist, output_p_travel = model(label_x)
            # output_dist = model(label_x)
            # output_x = model(train_x, label_d, label_p, label_s)
            # output_d, output_p, output_s = model(label_x)
            # output = model(label_x)
            #azimuth
            # output_azi = model(label_x)

            loss_dist = loss_mse(output_dist, label_dist)
            loss_p_travel = loss_mse(output_p_travel, label_p_travel)
            # loss_deep = loss_mse(output_deep, label_deep)
            # loss = loss_mse(output_azi, label_azimuth)

            #phase_net
            # loss = loss_func(output, phase_net_label)

            loss = loss_dist + loss_p_travel
            # loss = loss_d * 0.1 + loss_p + loss_s + loss_dist * 0.0001 + loss_p_travel * 0.1 + loss_deep * 0.001 + loss_azi
            # loss = loss_d * 0.05 + loss_p * 0.45 + loss_s * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            # running_results["pick_loss"] += (loss_d * 0.05 + loss_p * 0.45 + loss_s * 0.5).item() * batch_size
            # running_results["dist_loss"] += loss_dist.item() * batch_size
            # running_results["p_travel_loss"] += loss_p_travel.item() * batch_size
            # running_results["deep_loss"] += loss_deep.item() * batch_size
            # running_results["azi_loss"] += loss_azi.item() * batch_size

            train_bar.set_description(desc='[%d/%d] loss: %.7f' % (
                epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]
            ))
        train_loss.append(running_results["loss"] / running_results["batch_sizes"])
        train_pick_loss.append(running_results["pick_loss"] / running_results["batch_sizes"])
        train_p_travel_loss.append(running_results["p_travel_loss"] / running_results["batch_sizes"])
        train_dist_loss.append(running_results["dist_loss"] / running_results["batch_sizes"])
        train_deep_loss.append(running_results["deep_loss"] / running_results["batch_sizes"])
        train_azi_loss.append(running_results["azi_loss"] / running_results["batch_sizes"])

        # validation
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(range(len(validation)), colour="white")
            valing_results = {'batch_sizes': 0, "loss": 0, "pick_loss": 0, "p_travel_loss": 0, "dist_loss": 0, "deep_loss": 0, "azi_loss": 0}
            dist_error = []
            p_travel_error = []
            deep_error = []
            azi_error = []
            for i in val_bar:
                # validation_c, validation_d, validation_p, validation_s, validation_dist, validation_p_travel, validation_deep, validation_azimuth = validation.__getitem__(
                #     i)
                # val_baz_x = baz(c_shuffled[i], p_shuffled[i])
                # val_cov = cov(val_baz_x)
                # val_baz_x = torch.tensor(val_baz_x, dtype=torch.float32).permute(0, 2, 1)
                # val_cov = torch.tensor(val_cov, dtype=torch.float32)

                val_c = torch.tensor(validation_c[i], dtype=torch.float32).permute(0, 2, 1)
                # val_x = torch.tensor(val_noise_data[i], dtype=torch.float32).permute(0, 2, 1)
                # val_d = torch.tensor(validation_d[i], dtype=torch.float32).permute(0, 2, 1)
                # val_p = torch.tensor(validation_p[i], dtype=torch.float32).permute(0, 2, 1)
                # val_s = torch.tensor(validation_s[i], dtype=torch.float32).permute(0, 2, 1)
                val_dist = torch.tensor(validation_dist[i], dtype=torch.float32)
                val_p_travel = torch.tensor(validation_p_travel[i], dtype=torch.float32)
                # val_deep = torch.tensor(validation_deep[i], dtype=torch.float32)
                # val_azimuth = torch.tensor(validation_azimuth[i], dtype=torch.float32)

                # Phase_net
                # val_noise = torch.tensor(val_noise_prob[i], dtype=torch.float32).permute(0, 2, 1)
                # val_phase_net_label = torch.cat((val_noise, val_p, val_s), dim=1)

                # val_x_ssq = ssq_function(val_c)
                #
                # val_x_ssq = val_x_ssq.to(device)

                # val_phase_net_label = val_phase_net_label.to(device)
                # val_baz_x = val_baz_x.to(device)
                # val_cov = val_cov.to(device)

                val_c = val_c.to(device)
                # val_x = val_x.to(device)
                # val_d = val_d.to(device)
                # val_p = val_p.to(device)
                # val_s = val_s.to(device)
                val_dist = val_dist.to(device)
                val_p_travel = val_p_travel.to(device)
                # val_deep = val_deep.to(device)
                # val_azimuth = val_azimuth.to(device)
                batch_size = val_c.size(0)
                valing_results["batch_sizes"] += batch_size
                # output_x, output_d, output_p, output_s, output_dist, output_azi = model(val_x)
                output_dist, output_p_travel = model(val_c)
                # output_dist = model(val_c)
                # output_x = model(val_x, val_d, val_p, val_s)
                # output_d, output_p, output_s = model(val_c)
                # output = model(val_c)
                #azimuth
                # output_azi = model(val_c)

                dist_error.append((torch.sum(torch.abs(val_dist - output_dist))/128).item())
                p_travel_error.append((torch.sum(torch.abs(val_p_travel - output_p_travel))/128).item())
                # deep_error.append((torch.sum(torch.abs(val_deep - output_deep)) / 128).item())
                # azi_error.append((torch.sum(torch.abs((torch.atan2(val_azimuth[:, 1], val_azimuth[:, 0]) -
                #                                       torch.atan2(output_azi[:, 1], output_azi[:, 0])) * (180 / torch.pi))) / 128).item())

                # loss_x = loss_mse(output_x, val_c)
                loss_dist = loss_mse(output_dist, val_dist)
                loss_p_travel = loss_mse(output_p_travel, val_p_travel)
                # loss_deep = loss_mse(output_deep, val_deep)
                # loss = loss_mse(output_azi, val_azimuth)
                # loss = loss_mse(output_deep, val_deep)

                loss = loss_dist + loss_p_travel
                # loss = loss_d * 0.1 + loss_p + loss_s + loss_dist * 0.0001 + loss_p_travel * 0.1 + loss_deep * 0.001 + loss_azi
                # loss = loss_d * 0.05 + loss_p * 0.45 + loss_s * 0.5

                valing_results["loss"] += loss.item() * batch_size
                # valing_results["pick_loss"] += (loss_d * 0.05 + loss_p * 0.45 + loss_s * 0.5).item() * batch_size
                # valing_results["dist_loss"] += loss_dist.item() * batch_size
                # valing_results["p_travel_loss"] += loss_p_travel.item() * batch_size
                # valing_results["deep_loss"] += loss_deep.item() * batch_size
                # valing_results["azi_loss"] += loss_azi.item() * batch_size

                val_bar.set_description(desc='[val] loss: %.7f' % (
                        valing_results["loss"] / valing_results["batch_sizes"]))

        val_loss.append(valing_results["loss"] / valing_results["batch_sizes"])
        val_pick_loss.append(valing_results["pick_loss"] / valing_results["batch_sizes"])
        val_p_travel_loss.append(valing_results["p_travel_loss"] / valing_results["batch_sizes"])
        val_dist_loss.append(valing_results["dist_loss"] / valing_results["batch_sizes"])
        val_deep_loss.append(valing_results["deep_loss"] / valing_results["batch_sizes"])
        val_azi_loss.append(valing_results["azi_loss"] / valing_results["batch_sizes"])

        # plot the loss for each epoch
        if epoch > 0 and epoch % 10 == 0:
            fig1, ax1 = plt.subplots()
            ax1.plot(train_loss, color='#bb3f3f', label='Train_loss', marker='s', markersize=4)
            ax1.legend()
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            plt.show()

            fig1, ax1 = plt.subplots()
            ax1.plot(val_loss, color='#665fd1', label='Val_loss', marker='o', markersize=4)
            ax1.legend()
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            plt.show()

            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_pick_loss, color='#bb3f3f', label='Train_pick_loss', marker='s', markersize=4)
            # ax1.plot(val_pick_loss, color='#665fd1', label='Val_pick_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_dist_loss, color='#bb3f3f', label='Train_dist_loss', marker='s', markersize=4)
            # ax1.plot(val_dist_loss, color='#665fd1', label='Val_dist_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_p_travel_loss, color='#bb3f3f', label='Train_p_travel_loss', marker='s', markersize=4)
            # ax1.plot(val_p_travel_loss, color='#665fd1', label='Val_p_travel_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            # #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_deep_loss, color='#bb3f3f', label='Train_deep_loss', marker='s', markersize=4)
            # ax1.plot(val_deep_loss, color='#665fd1', label='Val_deep_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()
            #
            # fig1, ax1 = plt.subplots()
            # ax1.plot(train_azi_loss, color='#bb3f3f', label='Train_azi_loss', marker='s', markersize=4)
            # ax1.plot(val_azi_loss, color='#665fd1', label='Val_azi_loss', marker='o', markersize=4)
            # ax1.legend()
            # ax1.set_title('Training Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # plt.show()

        # update the LR
        scheduler.step()

        # save model parameters
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), model_path + "/detect_test_netpar_epoch_%d.pth" % epoch)

        metric = 1/(valing_results["loss"]/valing_results["batch_sizes"])
        if metric > save_metric:
            torch.save(model.state_dict(), model_path + "/detect_test_best.pth")
            save_metric = metric
            print("save new model")

        print('mean dist error', sum(dist_error)/len(validation_c))
        print('mean p travel time error', sum(p_travel_error) / len(validation_c))
        # print('mean deep error', sum(deep_error) / len(validation_c))
        # print('mean azi error', sum(azi_error) / len(validation_c))

    np.save(os.path.join(loss_path, 'loss_train'), train_loss)
    np.save(os.path.join(loss_path, 'loss_pick_train'), train_pick_loss)
    np.save(os.path.join(loss_path, 'loss_dist_train'), train_dist_loss)
    np.save(os.path.join(loss_path, 'loss_p_travel_train'), train_p_travel_loss)
    np.save(os.path.join(loss_path, 'loss_deep_train'), train_deep_loss)
    np.save(os.path.join(loss_path, 'loss_azi_train'), train_azi_loss)
    np.save(os.path.join(loss_path, 'loss_val'), val_loss)
    np.save(os.path.join(loss_path, 'loss_pick_val'), val_pick_loss)
    np.save(os.path.join(loss_path, 'loss_dist_val'), val_dist_loss)
    np.save(os.path.join(loss_path, 'loss_p_travel_val'), val_p_travel_loss)
    np.save(os.path.join(loss_path, 'loss_deep_val'), val_deep_loss)
    np.save(os.path.join(loss_path, 'loss_azi_val'), val_azi_loss)

    fig1, ax1 = plt.subplots()
    ax1.plot(train_loss, color='#bb3f3f', label='Train_loss', marker='s', markersize=4)
    ax1.plot(val_loss, color='#665fd1', label='Val_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(train_pick_loss, color='#bb3f3f', label='Train_pick_loss', marker='s', markersize=4)
    ax1.plot(val_pick_loss, color='#665fd1', label='Val_pick_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(train_dist_loss, color='#bb3f3f', label='Train_dist_loss', marker='s', markersize=4)
    ax1.plot(val_dist_loss, color='#665fd1', label='Val_dist_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(train_p_travel_loss, color='#bb3f3f', label='Train_p_travel_loss', marker='s', markersize=4)
    ax1.plot(val_p_travel_loss, color='#665fd1', label='Val_p_travel_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(train_deep_loss, color='#bb3f3f', label='Train_deep_loss', marker='s', markersize=4)
    ax1.plot(val_deep_loss, color='#665fd1', label='Val_deep_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(train_azi_loss, color='#bb3f3f', label='Train_azi_loss', marker='s', markersize=4)
    ax1.plot(val_azi_loss, color='#665fd1', label='Val_azi_loss', marker='o', markersize=4)
    ax1.legend()
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.show()
    print(f'min loss in train: {min(train_loss):.7f}')
    print(f'min loss in validation: {min(val_loss):.7f}')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print('running time: %dmin' % elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super Resolution AND Denoising Model")
    parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
    parser.add_argument("--lr", default=0.0005, type=float, help="learning rate of model")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size of train dataset")
    parser.add_argument("--out_path", default="./Project2", type=str, help="the path of save file")
    opt = parser.parse_args()
    main(opt)
