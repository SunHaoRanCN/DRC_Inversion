from src.models.DECOMP import compressor, decompressor
from utiles.audio_loss import loss_mse, MelSTFTLoss
import time
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec


def get_array(x, threshold):
    delta_x = x * threshold
    x_min = x - delta_x
    x_max = x + delta_x
    return np.linspace(x_min, x_max, 11)

compressors = {
    'A': {'param1': -32, 'param2': 3, 'param3': 5, 'param4': 5, 'param5': 13, 'param6': 435, 'param7': 2},
    'B': {'param1': -19.9, 'param2': 1.8, 'param3': 5, 'param4': 5, 'param5': 11, 'param6': 49, 'param7': 2},
    'C': {'param1': -24.4, 'param2': 3.2, 'param3': 5, 'param4': 5, 'param5': 5.8, 'param6': 112, 'param7': 2},
    'D': {'param1': -28.3, 'param2': 7.3, 'param3': 5, 'param4': 5, 'param5': 9, 'param6': 705, 'param7': 2},
    'E': {'param1': -38, 'param2': 4.9, 'param3': 5, 'param4': 5, 'param5': 3.1, 'param6': 257, 'param7': 2},
}

# compressors = pickle.load(open(r"30profiles.pkl", 'rb'))

# duration = 0.5  # seconds
# fs = 44100  # Hz
# t = np.linspace(0, duration, int(fs * duration), endpoint=False)
#
# piecewise_func = lambda x: 0 if 0 <= x < 0.1 else \
#     0.25 if 0.1 <= x < 0.2 else \
#         -1.0 if 0.2 <= x < 0.3 else \
#             0.5 if 0.3 <= x < 0.4 else 0
#
# audio = np.zeros(len(t))
# for i in range(len(t)):
#     audio[i] = piecewise_func(t[i])
# x = audio

audio_path = '/home/hsun/Datasets/MedleyDB/5profiles/20dB/all//1_1_0.wav'
x, fs = sf.read(audio_path)
x = x[int(1.5*fs):int(2.5*fs)]
x = x[0:fs]


parameter_names = ['L', 'R', r'$\tau_{v}^{att}$', r'$\tau_{v}^{rel}$', r'$\tau_{g}^{att}$', r'$\tau_{g}^{rel}$']
variation_range = (-0.5, 0.5)  # -50% to +50%
n_variations = 10
variation_factors = np.linspace(variation_range[0], variation_range[1], n_variations)

Mel = MelSTFTLoss()

MSE_results = []
MEL_results = []
# modified_signal = []

labels = ['A', 'B', 'C', 'D', 'E']
# labels = [str(i) for i in range(1, 31)]
#
# compressed_signal = [x]
#
# for i in range(len(compressors)):
#     config_idx = labels[i]
#     print(f"Processing profile {i + 1}/{len(compressors)}")
#
#     original_params = list(compressors[config_idx].values())
#     original_params = np.array(original_params)
#
#     # Get original compressed and decompressed signals
#     y = compressor(x, 44100, original_params[0], original_params[1], original_params[2], original_params[3],
#                    original_params[4], original_params[5], 2)
# #     compressed_signal.append(y)
# #
# # np.savetxt("compressed_signal.txt", compressed_signal)
#
#     x_hat = decompressor(y, 44100, original_params[0], original_params[1], original_params[2], original_params[3],
#                          original_params[4], original_params[5], 2)
#
#     mse_results = np.zeros((6, 10))
#     mel_results = np.zeros((6, 10))
#     # For each parameter
#     for param_idx in range(6):
#         for var_idx in range(10):
#             print(i, param_idx, var_idx)
#
#             modified_params = original_params.copy()
#             modified_params[param_idx] *= (1 + variation_factors[var_idx])
#
#             # Get new reconstruction
#             x_hat_modified = decompressor(y, 44100, modified_params[0], modified_params[1], modified_params[2], modified_params[3],
#                                           modified_params[4], modified_params[5], 2)
#             # modified_signal.append(x_hat_modified)
#
#             # # Calculate MSE
#             mse = loss_mse(x_hat_modified, x_hat)
#             mse_results[param_idx][var_idx] = mse
#
#             # Calculate Mel
#             mel = Mel.compute_loss(x_hat_modified, x_hat)
#             mel_results[param_idx][var_idx] = mel
#
#     MSE_results.append(mse_results)
#     MEL_results.append(mel_results)
#
# arr_mse_results = np.stack(MSE_results).reshape(6*len(compressors), 10)
# np.savetxt("imp_mse_5.txt", arr_mse_results)
#
# arr_mel_results = np.stack(MEL_results).reshape(6*len(compressors), 10)
# np.savetxt("imp_mel_5.txt", arr_mel_results)

arr_mse_results = np.loadtxt("../results/mse_results.txt")
arr_mel_results = np.loadtxt("../results/mel_results.txt")

num_groups = len(compressors)
boxes_per_group = 6
group_labels = np.arange(1, num_groups + 1)

x_positions = np.arange(6*len(compressors))

grouped_metrix = [[] for _ in range(6)]

for i, arr in enumerate(arr_mel_results):
    group_index = i % 6
    grouped_metrix[group_index].append(arr)


boxplot_data = [np.array(group).flatten() for group in grouped_metrix]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                              gridspec_kw={
                                  'height_ratios': [1, 3],
                                  'hspace': 0.05
                              },
                              figsize=(12, 8))

# 共享x轴并隐藏中间边框
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.tick_params(axis='x', which='both', top=False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 绘制箱线图（分开展示）
box1 = ax1.boxplot(boxplot_data,
                  tick_labels=parameter_names,
                  showfliers=False,
                  patch_artist=True)

box2 = ax2.boxplot(boxplot_data,
                  tick_labels=parameter_names,
                  showfliers=False,
                  patch_artist=True)

ax1.set_ylim(1.5, 5)
ax2.set_ylim(-0.02, 1.1)

# ax1.set_ylim(0.28, 0.34)
# ax2.set_ylim(-0.0002, 0.007)

d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

all_medians = [np.median(data) for data in boxplot_data]
norm = plt.Normalize(vmin=np.min(all_medians), vmax=np.max(all_medians))
cmap = plt.cm.Reds

for i, box in enumerate(box2['boxes']):
    color = cmap(norm(all_medians[i]))
    box.set_facecolor(color)
    box.set_edgecolor('black')

for i, box in enumerate(box1['boxes']):
    color = cmap(norm(all_medians[i]))
    box.set_facecolor(color)
    box.set_edgecolor('black')

# 设置标签和样式
ax2.set_xticks(np.arange(len(boxplot_data))+1)
ax2.set_xticklabels(parameter_names, size=18)
ax2.set_xlabel('Parameter', fontsize=18)
# ax1.set_ylabel(r'$\mathcal{L}^{\text{Mel}}_{\hat{x},x}$', fontsize=18)
# ax2.set_ylabel(r'$\mathcal{L}^{\text{Mel}}_{\hat{x},x}$', fontsize=18)
fig.text(0.05, 0.5,
        r'$\mathcal{L}^{\text{Mel}}_{\hat{x},x}$',
        va='center',
        ha='center',
        rotation='vertical',
        fontsize=18)
ax1.tick_params(axis='y', labelsize=14)  # 上方坐标系
ax2.tick_params(axis='y', labelsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(pad=0.1)
plt.savefig("../Figures/imp_mel.pdf", format="pdf", transparent=True, bbox_inches='tight')
plt.show(block=True)

# plt.figure(figsize=(12, 6))
# boxplot = plt.boxplot(boxplot_data,
#                       labels=parameter_names,
#                       showfliers=False,
#                       patch_artist=True)
#
# medians = [median.get_ydata()[0] for median in boxplot['medians']]
#
# min_median = np.min(medians)
# max_median = np.max(medians)
# norm = plt.Normalize(vmin=min_median, vmax=max_median)
#
# cmap = plt.cm.Reds
#
# for i, (box, median_val) in enumerate(zip(boxplot['boxes'], medians)):
#     color = cmap(norm(median_val))
#     box.set_facecolor(color)
#     box.set_edgecolor('black')
#
# plt.xlabel('Parameter', fontsize=18)
# # plt.ylabel(r'$\mathcal{L}^{\text{MSE}}_{\hat{x},x}$', fontsize=18)
# plt.ylabel(r'$\mathcal{L}^{\text{Mel}}_{\hat{x},x}$', fontsize=18)
# plt.xticks(size=16)
# # plt.yscale("log")
# # plt.ylim([0, 1])
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show(block=True)


########### Calculate and print statistics
# print("\nMel Statistics for each parameter:")
# print("Parameter | Mean Mel | Std Dev")
# print("-" * 40)
# for i in range(6):
#     mean_mse = np.mean(results[i])
#     std_mse = np.std(results[i])
#     print(f"Param {i+1:4d} | {mean_mse:.6f} | {std_mse:.6f}")


# label = 'C'
# parameters = compressors[label]
# tic = time.time()
# y, g, v, f = compressor(audio, 44100, parameters.get('param1'), parameters.get('param2'), parameters.get('param3'),
#                         parameters.get('param4'), parameters.get('param5'), parameters.get('param6'),
#                         parameters.get('param7'))
# duration = time.time() - tic
# print(duration)

# threshold_range = get_array(parameters.get('param1'), 0.5)
# E_th = []
# for i in threshold_range:
#     x_hat, gg, vv, x2 = decompressor(y, 44100, i, parameters.get('param2'), parameters.get('param3'),
#                                      parameters.get('param4'), parameters.get('param5'), parameters.get('param6'),
#                                      parameters.get('param7'))
#     Error = rmse(audio, x_hat)
#     E_th.append(Error)
#
# ratio_range = get_array(parameters.get('param2'), 0.5)
# E_r = []
# for i in ratio_range:
#     x_hat, gg, vv, x2 = decompressor(y, 44100, parameters.get('param1'), i, parameters.get('param3'),
#                                      parameters.get('param4'), parameters.get('param5'), parameters.get('param6'),
#                                      parameters.get('param7'))
#     Error = rmse(audio, x_hat)
#     E_r.append(Error)
#
# tgatt_range = get_array(parameters.get('param3'), 0.5)
# E_tgatt = []
# for i in tgatt_range:
#     x_hat, gg, vv, x2 = decompressor(y, 44100, parameters.get('param1'), parameters.get('param2'), i,
#                                      parameters.get('param4'), parameters.get('param5'), parameters.get('param6'),
#                                      parameters.get('param7'))
#     Error = rmse(audio, x_hat)
#     E_tgatt.append(Error)
#
# tgrel_range = get_array(parameters.get('param4'), 0.5)
# E_tgrel = []
# for i in tgrel_range:
#     x_hat, gg, vv, x2 = decompressor(y, 44100, parameters.get('param1'), parameters.get('param2'),
#                                      parameters.get('param3'),
#                                      i, parameters.get('param5'), parameters.get('param6'),
#                                      parameters.get('param7'))
#     Error = rmse(audio, x_hat)
#     E_tgrel.append(Error)
#
# tvatt_range = get_array(parameters.get('param5'), 0.5)
# E_tvatt = []
# for i in tvatt_range:
#     x_hat, gg, vv, x2 = decompressor(y, 44100, parameters.get('param1'), parameters.get('param2'),
#                                      parameters.get('param3'),
#                                      parameters.get('param4'), i, parameters.get('param6'),
#                                      parameters.get('param7'))
#     Error = rmse(audio, x_hat)
#     E_tvatt.append(Error)
#
# tvrel_range = get_array(parameters.get('param6'), 0.5)
# E_tvrel = []
# for i in tvrel_range:
#     x_hat, gg, vv, x2 = decompressor(y, 44100, parameters.get('param1'), parameters.get('param2'),
#                                      parameters.get('param3'),
#                                      parameters.get('param4'), parameters.get('param5'), i,
#                                      parameters.get('param7'))
#     Error = rmse(audio, x_hat)
#     E_tvrel.append(Error)
#
# xticks = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])  # Tick positions from -0.5 to 0.5
# xtick_labels = [f'{tick}%' for tick in xticks]
#
# plt.figure()
# plt.plot(xticks, E_th, marker='o', label='threshold')
# plt.plot(xticks, E_r, marker='p', label='ratio')
# plt.plot(xticks, E_tgatt, marker='v', label=r'$\tau_{g,att}$')
# plt.plot(xticks, E_tgrel, marker='s', label=r'$\tau_{g,rel}$')
# plt.plot(xticks, E_tvatt, marker='d', label=r'$\tau_{v,att}$')
# plt.plot(xticks, E_tvrel, marker='*', label=r'$\tau_{v,rel}$')
# plt.axvline(x=0, color='red', linestyle='--', label='real value')
# plt.legend()
# plt.xlabel('Parameter error (%)')
# plt.xticks(xticks, xtick_labels)
# plt.ylabel('RMSE (dB)')
# plt.show()
