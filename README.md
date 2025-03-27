# SSDPL: A completely deep neural network-based method for single-station seismic detection, phase picking, and epicenter location
## This method utilizes a neural network model for the detection and localization of seismic events from a single station. 
You can leverage our pre-trained models for event detection and phase picking on actual seismic records. Additionally, we offer a single-station location method. Our pre-trained models demonstrate strong performance, particularly for events with smaller magnitudes. Furthermore, you have the option to retrain these models with your own data to better align with your specific task requirements.
> [!NOTE]
>The code will be continuously maintained and updated. If you encounter any issues, please contact me📧.
> 
> The code is built with PyTorch.
> 
> Please install the necessary Python Package before executing the code.
# How to check the seismic waveforms and labels of trainning sets？
This is a code example demonstrating how to plot waveforms from an augmented training dataset.

You can download the [STanford EArthquake Dataset (STEAD) here!](https://github.com/smousavi05/STEAD?tab=readme-ov-file)
```python
import numpy as np
import matplotlib.pyplot as plt
from Read_data import read
from matplotlib.ticker import FormatStrFormatter

file_name = "../STEAD/merged/merge.hdf5"
csv_file = "../STEAD/merged/merge.csv"

training_data = read(input_hdf5=file_name, input_csv=csv_file, batch_size=128, augmentation=True, mode='test')
batch_x, batch_d, batch_p, batch_s, batch_dist, batch_p_travel, batch_deep, batch_azi = training_data.__getitem__(1)

for k in range(0, 63):
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.1)

    axs[0].plot(batch_x[k, 0, :], 'black', label='E component')
    axs[0].set_xticklabels([])
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[1].plot(batch_x[k, 1, :], 'black', label='N component')
    axs[1].set_xticklabels([])
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[2].plot(batch_x[k, 2, :], 'black', label=' component')
    axs[2].set_xticklabels([])
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[3].plot(batch_d[k, 0, :], '#65ab7c', linestyle='--', linewidth=2, label='Label event')
    axs[3].plot(batch_p[k, 0, :], '#665fd1', linestyle='--', linewidth=2, label='Label p')
    axs[3].plot(batch_s[k, 0, :], '#c44240', linestyle='--', linewidth=2, label='Label s')
    axs[3].set_ylabel('Probability')
    axs[3].set_ylim(0, 1.1)
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')

    labels = ['(I)', '(II)', '(III)']
    for ax, label in zip(axs.flat, labels):
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1))

    plt.show()
```
![image](https://github.com/Guo-my/SSDPL/blob/main/Figure/waveforms.png)

# How to Test Model Generalization Using Japanese Strong Motion Data (K-NET)？
Since the STEAD training data does not encompass earthquake data from Japan, utilizing strong motion data from K-Net to evaluate the performance of the network model is a good choice. You can access [K-NET data here](https://www.kyoshin.bosai.go.jp/).
This code shows a function how to read the K-Net data.
```python
def read_knet_ascii(file_path):
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

        metadata = {}
        for line in lines[:17]:
            parts = line.strip().split()
            if len(parts) >= 4:
                metadata[parts[0] + parts[1]] = parts[2] + parts[3]
            if len(parts) == 3:
                metadata[parts[0] + parts[1]] = parts[-1]
            if len(parts) == 2:
                metadata[parts[0]] = parts[-1]
        data['metadata'] = metadata

        scale_line = lines[13]
        parts = scale_line.strip().split()
        numerator, denominator = parts[-1].split('/')
        numerator = numerator.replace('(gal)', '')

        numerator = float(numerator)
        denominator = float(denominator)
        proportion = numerator / denominator

        waveform_data = []
        for line in lines[17:]:
            parts = line.strip().split()
            if parts:
                try:
                    for part in parts:
                        waveform_data.append((float(part)*proportion))
                except ValueError:
                    continue
        data['waveform'] = waveform_data

    return data
```
![image](https://github.com/user-attachments/assets/f37a7836-54aa-4506-a0af-9108d09b8570)
