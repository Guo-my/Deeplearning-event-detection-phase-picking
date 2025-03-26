import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def calculate_error(true, pre, threshold):
    """

    Calculate MAE and Std.

    Parameters
    ----------
    true : 2D array
        Ground truth labels.

    pre : 3D array
        Predicted labels.

    Returns
    -------
    m_error, Sigma, Mae, Mape, plot: float

    """
    min_error = 0
    sigma_error = 0
    mae_error = 0
    mape_error = 0
    plot = []
    num = 0
    for i in range(pre.shape[0]):
        error = []
        sigma = []
        mae = []
        mape = []
        pre_prob = pre[i, 0, :]
        true_ground_prob = true[i, :]
        if all(x == 0 for x in true_ground_prob):
            continue
        # max_value = pre_prob * (pre_prob > threshold)
        indexes = np.where(pre_prob >= threshold)[0]
        if indexes.size == 0:
            continue
        max_index_within_threshold = indexes[np.argmax(pre_prob[indexes])]
        indexes_true = float(np.where(true_ground_prob == 1)[0])
        min_error += (indexes_true - max_index_within_threshold) * 0.01
        sigma_error += ((indexes_true - max_index_within_threshold) * 0.01) ** 2
        mae_error += abs(indexes_true - max_index_within_threshold) * 0.01
        mape_error += abs(indexes_true - max_index_within_threshold) / indexes_true
        # for indexes_value in indexes:
        #     error.append((indexes_true - indexes_value) * 0.01)
        # for indexes_value in indexes:
        #     sigma.append(((indexes_true - indexes_value) * 0.01) ** 2)
        # for indexes_value in indexes:
        #     mae.append(abs(indexes_true - indexes_value) * 0.01)
        # for indexes_value in indexes:
        #     mape.append(abs(indexes_true - indexes_value) / indexes_true)

        plot.append((indexes_true - max_index_within_threshold) * 0.01)

        # min_error += min(error, key=abs)
        # sigma_error += min(sigma)
        # mae_error += min(mae)
        # mape_error += min(mape)
        num += 1
    m_error = min_error / num
    Sigma = math.sqrt(sigma_error / num)
    Mae = mae_error / num
    Mape = mape_error * 0.01 / num
    return m_error, Sigma, Mae, Mape, plot


if __name__ == "__main__":
    true_d = np.load("../new_file/MODEL_TRAIN/test_d_new.npy")
    true_p = np.load("../new_file/MODEL_TRAIN/test_p_new.npy")
    true_s = np.load("../new_file/MODEL_TRAIN/test_s_new.npy")
    d = np.load("./OUTPUT/pre_result/pre_detect_test_raw.npy")
    p = np.load("./OUTPUT/pre_result/pre_p_test_pywt.npy")
    s = np.load("./OUTPUT/pre_result/pre_s_test_pywt.npy")
    print(true_d.shape, d.shape)

    mean_error_p, sigma_p, mae_p, mape_p = calculate_error(true_p, p, 0.2)
    mean_error_s, sigma_s, mae_s, mape_s = calculate_error(true_s, s, 0.2)
    print(mean_error_p, sigma_p, mae_p, mape_p)
    print(mean_error_s, sigma_s, mae_s, mape_s)

    x_data = np.linspace(0, 60, 6000)
    for j in range(100):
        fig, axs = plt.subplots(2, 1, figsize=(15, 20))
        fig.subplots_adjust(hspace=0.3)

        axs[0].plot(x_data, true_d[j, :], 'green', label='Pre_detector_event')
        axs[0].plot(x_data, true_p[j, :], 'blue', label='Pre_pick_P_wave', linestyle='--')
        axs[0].plot(x_data, true_s[j, :], 'yellow', label='Pre_pick_S_wave', linestyle='--')
        axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[0].set_ylabel('Probability')
        axs[0].legend()

        axs[1].plot(x_data, d[j, 0, :], 'green', label='Pre_detector_event')
        axs[1].plot(x_data, p[j, 0, :], 'blue', label='Pre_pick_P_wave', linestyle='--')
        axs[1].plot(x_data, s[j, 0, :], 'yellow', label='Pre_pick_S_wave', linestyle='--')
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1].set_ylabel('Probability')
        axs[1].legend()

        # axs[1].plot(x_data, true_d[j+35, :, 0], 'green', label='Detector_event')
        # axs[1].plot(x_data, true_p[j+35, :, 0], 'blue', label='Pick_P_wave', linestyle='--')
        # axs[1].plot(x_data, true_s[j+35, :, 0], 'yellow', label='Pick_S_wave', linestyle='--')
        # axs[1].set_xticklabels([])
        # axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[1].set_ylabel('Probability')
        # axs[1].legend()
        #
        # axs[2].plot(x_data, pre_d_result[j+35, 0, :], 'green', label='Pre_detector_event')
        # axs[2].plot(x_data, pre_p_result[j+35, 0, :], 'blue', label='Pre_pick_P_wave', linestyle='--')
        # axs[2].plot(x_data, pre_s_result[j+35, 0, :], 'yellow', label='Pre_pick_S_wave', linestyle='--')
        # axs[2].set_xticklabels([])
        # axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[2].set_ylabel('Probability')
        # axs[2].legend()

        labels = ['(I)', '(II)', '(III)']
        for ax, label in zip(axs.flat, labels):
            ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1))
        fig.text(0.5, 0.08, 'Time(s)', ha='center', va='center', fontsize=20)
        plt.show()

        inp = input("Press a key to plot the next waveform!")
        if inp == "r":
            continue