import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xlrd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import colorsys
import matplotlib.colors as mcolors
from adjustText import adjust_text
import pickle
from scipy import signal

#####################
# FC layers and acc #
#####################
### input the results manually
datasets = {
    "MedleyDB-31": (np.array([0.831, 0.842, 0.854, 0.832, 0.822]), np.array([0.0367, 0.0372, 0.0380, 0.0383, 0.0388])),
    "Mus-31": (np.array([0.815, 0.828, 0.843, 0.820, 0.805]), np.array([0.0367, 0.0372, 0.0380, 0.0383, 0.0388])),
    "Dafx-31": (np.array([0.862, 0.893, 0.884, 0.874, 0.832]), np.array([0.0356, 0.0361, 0.0370, 0.0372, 0.0375])),
    "Libri-31": (np.array([0.892, 0.923, 0.903, 0.889, 0.876]), np.array([0.0351, 0.0355, 0.0360, 0.0363, 0.0368]))
}

layers = np.array([3, 4, 5, 6, 7])

fig, axs = plt.subplots(4, 1, figsize=(12, 8))
plt.subplots_adjust(hspace=0.4)

bar_width = 0.5 
acc_color = '#1f77b4'
time_color = '#ff7f0e'

for idx, (title, (acc, time)) in enumerate(datasets.items()):
    ax = axs[idx]
    ax2 = ax.twinx()

    bars = ax2.bar(layers, time, bar_width,
                   color=time_color, alpha=0.3,
                   # label='Time/epoch',
                   label='Temps/époque',
                   align='center') 
    ax2.set_ylabel('Temps (h)', fontsize=18)
    ax2.tick_params(axis='y', labelcolor=time_color)
    ax2.set_ylim(0.034, 0.039)

    line = ax.plot(layers, acc, color=acc_color, marker='o',
                   linewidth=2, markersize=8,
                   # label='Accuracy',
                   label='Précision'
                   )
    ax.set_ylabel('Précision', fontsize=18)
    ax.tick_params(axis='y', labelcolor=acc_color)
    ax.set_ylim(0.8, 0.945)

    ax.set_title(title, fontsize=18, pad=10)
    ax.set_xlabel('Nombre de couches PMC', fontsize=18) if idx == 3 else None    # Nombre de couches PMC / Number of FC layer
    ax.set_xticks(layers)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    handles = line + [bars]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc='upper left', ncol=2, frameon=False)

plt.tight_layout()
plt.savefig("../Figures/layer_large_fr.pdf", format="pdf", transparent=True)
plt.show(block=True)


########################
# R2 metrice and score #
########################

def process_q(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Convert string representations of arrays to actual arrays
    def parse_array(arr_str):
        # Remove brackets and split by spaces
        return np.array([float(x) for x in arr_str.strip('[]').split()])

    # Convert the string arrays to numerical arrays
    real_arrays = df['real'].apply(parse_array)
    estimated_arrays = df['estimated'].apply(parse_array)

    # Initialize lists to store separated parameters
    real_params = [[] for _ in range(6)]
    estimated_params = [[] for _ in range(6)]

    for real_arr, est_arr in zip(real_arrays, estimated_arrays):
        for i in range(6):
            real_params[i].append(real_arr[i])
            estimated_params[i].append(est_arr[i])

    # Convert lists to numpy arrays
    real_params = [np.array(param) for param in real_params]
    estimated_params = [np.array(param) for param in estimated_params]

    return real_params, estimated_params


param_names = ['L', 'R', r'$\tau_v^{att}$', r'$\tau_v^{rel}$', r'$\tau_g^{att}$', r'$\tau_g^{rel}$']

file_path = 'parameter.xlsx'  # path to the parameter table file
real_parameters, estimated_parameters = process_q(file_path)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes_flat = axes.flatten()

# Create plots for each parameter
for i in range(6):
    ax = axes_flat[i]

    # Get real and estimated values for current parameter
    real = real_parameters[i]
    est = estimated_parameters[i]

    # Calculate R² scores and fit lines for train set
    reg_train = LinearRegression()
    Est = est.reshape(-1, 1)
    reg_train.fit(Est, real)
    r2 = r2_score(real, est)

    ax.scatter(est, real, s=5)

    x_train_range = np.linspace(min(est), max(est), 100)
    y_train_pred = reg_train.predict(x_train_range.reshape(-1, 1))
    ax.plot(x_train_range, y_train_pred, color='red', linestyle='--', label='fitted line')    # droite ajustée / fitted line
    ax.plot(x_train_range, x_train_range, color='green', linestyle='--', label='ref line')   # ligne de réf / ref line

    if i >= 3:  # Bottom row
        ax.set_xlabel('Estimated Parameter', fontsize=18)
    else:
        ax.set_xticklabels([])  # Remove x-tick labels for top row

    if i % 3 == 0:  # First column
        ax.set_ylabel('Real Parameter', fontsize=18)
    else:
        ax.set_yticklabels([])

    ax.set_title(f'{param_names[i]} (R²: {r2:.3f})', fontsize=18)

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('auto')

plt.tight_layout()
plt.savefig("R2.pdf", format="pdf", transparent=True)
plt.show(block=True)



############
# baseline #
############

models = set()
datasets = set()

for file in glob.glob('loss_*.xlsx'):
    filename = os.path.splitext(os.path.basename(file))[0]
    parts = filename.split('_')
    
    if len(parts) < 3 or parts[0] != 'loss':
        continue
    
    models.add(parts[1])
    datasets.add('_'.join(parts[2:]))

for dataset in datasets:
    globals()[f"MSE_{dataset}"] = {}
    globals()[f"Mel_{dataset}"] = {}
    globals()[f"SISDR_{dataset}"] = {}

for file in glob.glob('loss_*.xlsx'):
    filename = os.path.splitext(os.path.basename(file))[0]
    parts = filename.split('_')
    
    if len(parts) < 3 or parts[0] != 'loss':
        continue
    
    model = parts[1]
    dataset = '_'.join(parts[2:])
    
    try:
        df = pd.read_excel(file, header=None)

        mse = df[0].tolist()
        mel = df[1].tolist()
        sisdr = df[2].tolist()
        
        globals()[f"MSE_{dataset}"][model] = mse
        globals()[f"Mel_{dataset}"][model] = mel
        globals()[f"SISDR_{dataset}"][model] = sisdr
    except Exception as e:
        print(f"Error while reading file: {file} !")

data = {
    'MedleyDB': {'MSE': MSE_MedleyDB, 'Mel': Mel_MedleyDB, 'SISDR': SISDR_MedleyDB},
    'MUSDB18-HQ': {'MSE': MSE_musdb18hq, 'Mel': Mel_musdb18hq, 'SISDR': SISDR_musdb18hq},
    'DAFX': {'MSE': MSE_DAFX, 'Mel': Mel_DAFX, 'SISDR': SISDR_DAFX},
    'LibriSpeech': {'MSE': MSE_LibriSpeech, 'Mel': Mel_LibriSpeech, 'SISDR': SISDR_LibriSpeech}
}


fig, axes = plt.subplots(4, 3, figsize=(16, 8))


row_datasets = ['MedleyDB', 'MUSDB18-HQ', 'DAFX', 'LibriSpeech']
xlims = {'MSE': (0, 0.23), 'Mel': (0, 1.75), 'SISDR': (-15, 40)}

col_metrics = ['MSE', 'Mel', 'SISDR']
# col_metrics = [
#     r'$\mathcal{L}^{\mathrm{MSE}}_{\hat{x},x}$',
#     r'$\mathcal{L}^{\mathrm{Mel}}_{\hat{x},x}$',
#     r'$\mathrm{SISDR}$'
# ]

# Colors for each model
colors = ['#7293CB', '#68AD7C', '#CD6666', '#937AB6', '#7DABB8', '#C76B6B']

# Create box plots for each subplot
for row, dataset in enumerate(row_datasets):
    for col, metric in enumerate(col_metrics):
        ax = axes[row, col]
        current_data = data[dataset][metric]

        positions = np.arange(len(current_data), 0, -1)

        for (name, values), pos, color in zip(current_data.items(), positions, colors):
            if name == 'Ref':
                cilo_factor = 0.995
                cihi_factor = 1.005
            else:
                cilo_factor = 0.95
                cihi_factor = 1.05

            stats = {
                'MedleyDB': values[2],
                'cilo': values[2] * cilo_factor,
                'cihi': values[2] * cihi_factor,
                'q1': values[1],
                'q3': values[3],
                'whislo': values[0],
                'whishi': values[4]
            }

            box = ax.bxp([stats],
                         positions=[pos],
                         vert=False,
                         widths=0.7,
                         patch_artist=True,
                         showfliers=False)

            plt.setp(box['boxes'], facecolor=color, alpha=0.7)
            plt.setp(box['MedleyDBians'], color='black')
            plt.setp(box['whiskers'], color='black')
            plt.setp(box['caps'], color='black')

        ax.set_xlim(xlims[metric])
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.axvline(x=xlims[metric][0], color='gray', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_yticks(positions)
        if col == 0:
            ax.set_yticklabels(list(current_data.keys()))
        else:
            ax.set_yticklabels([])

        if row == 3:
            ax.set_xlabel(metric, fontsize=18)
        if row != 3:
            ax.set_xticklabels([])
        if col == 2:
            ax.text(1.05, 0.5, dataset,
                    transform=ax.transAxes,
                    rotation=-90,
                    verticalalignment='center',
                    fontsize=18)
plt.tight_layout()
plt.savefig("baseline.pdf", format="pdf", transparent=True)
plt.show(block=True)


