import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.AudioDataset import get_2D_dataset
import os
from models.ast_model import ASTModel
from models.DECOMP import decompressor
import time
import auraloss
import pickle
import librosa
import soundfile as sf
from utiles.utile import set_seed


def prediction_to_label(predictions):
    p = []
    for pre_label in predictions:
        l = str(pre_label.cpu().numpy())
        p.append(l)
    return p


def normSignal(x):
    x = x - torch.mean(x, dim=1, keepdim=True)
    x = x / torch.sqrt(torch.mean(torch.square(x), dim=1, keepdim=True))
    return x

def normSignal_np(x):
    x = x - np.mean(x)
    x = x / np.sqrt(np.mean(x**2))
    return x


# Hyperparameters
seed = 11   # 11 -> MLP, 22 -> KAN
set_seed(seed)
batch_size = 12
lr = 0.00001
n_epochs = 200
norm_param = 0.01
target_shape = (64, 431)

### Data loading and preprocessing
ref_folder = "/home/hsun/Datasets/LibriSpeech/5profiles/clean/all"
train_folder = "/home/hsun/Datasets/LibriSpeech/5profiles/clean/train"
test_folder = "/home/hsun/Datasets/LibriSpeech/5profiles/clean/test"
eval_folder = test_folder
output_folder = " "

train_dataset = get_2D_dataset(train_folder, target_shape=target_shape)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = get_2D_dataset(test_folder, target_shape=target_shape)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

audio, raw_audio, feature, compressor_type, audio_name = train_dataset[0]
f_dim = feature.shape[0]
t_dim = feature.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTModel(label_dim=6, input_tdim=target_shape[1], input_fdim=target_shape[0], imagenet_pretrain=True, audioset_pretrain=False)
model = nn.DataParallel(model)
model = model.to(device)
# summary(model, estimate_size = (batch_size, 1, 220500), device = device)

# compressors = pickle.load(open(r"30profiles_RMS.pkl", 'rb'))
compressors = {
    'A': {'param1': -32, 'param2': 3, 'param3': 5, 'param4': 5, 'param5': 13, 'param6': 435, 'param7': 2},
    'B': {'param1': -19.9, 'param2': 1.8, 'param3': 5, 'param4': 5, 'param5': 11, 'param6': 49, 'param7': 2},
    'C': {'param1': -24.4, 'param2': 3.2, 'param3': 5, 'param4': 5, 'param5': 5.8, 'param6': 112, 'param7': 2},
    'D': {'param1': -28.3, 'param2': 7.3, 'param3': 5, 'param4': 5, 'param5': 9, 'param6': 705, 'param7': 2},
    'E': {'param1': -38, 'param2': 4.9, 'param3': 5, 'param4': 5, 'param5': 3.1, 'param6': 257, 'param7': 2},
}
compressors_torch = {key: {param: torch.tensor(value).to(device) for param, value in params.items()} for key, params
                     in compressors.items()}

MSE = torch.nn.MSELoss()
Mel = auraloss.freq.MelSTFTLoss(sample_rate=22050, fft_size=2048, hop_size=512, win_length=2048, n_mels=128, device=device, w_sc=0).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# Training loop with early stopping
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Early stopping parameters
min_loss_val = 10
patience = 30  # Number of epochs with no improvement after which training will be stopped
best_val_loss = float('inf')
patience_counter = 0
best_test_acc = 0

early_stop = True
if early_stop:
    print("Early stop with patience of {} epochs".format(patience))
else:
    print("Training for {} epochs without early stop".format(n_epochs))

torch.cuda.synchronize()
start = time.time()

for epoch in range(n_epochs):
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    model.train()
    running_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0
    all_labels_train = []
    all_predictions_train = []

    for real_signals, target_signals, inputs, labels, audio_name in train_loader:
        inputs = inputs.to(torch.float32)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples_train += labels.size(0)
        correct_predictions_train += (predicted == labels).sum().item()

        all_labels_train.extend(labels.cpu().numpy())
        all_predictions_train.extend(predicted.cpu().numpy())
    scheduler.step()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions_train / total_samples_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    ### Validation
    model.eval()
    val_loss = 0.0
    correct_predictions_val = 0
    total_samples_val = 0
    all_labels = []
    all_predictions = []
    count = 0

    with torch.no_grad():
        for real_signals, target_signals, inputs, labels, audio_name in test_loader:
            inputs = inputs.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total_samples_val += labels.size(0)
            correct_predictions_val += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(test_loader)
        val_accuracy = correct_predictions_val / total_samples_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1:03d} | Current lr: {learning_rate:.7f} \n Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}')

    if val_loss <= min_loss_val:
        min_loss_val = val_loss
        best_model = model.state_dict()
        torch.save(best_model, '../../pretrained models/LibriSpeech/ast_libri_1024.pt')

    # Early stopping check
    if epoch < 30:
        continue
    else:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping after {epoch + 1} epochs without improvement.')
        break

torch.cuda.synchronize()
end = time.time()

training_time = end - start
print(f'Training Time : {round(training_time / 3600, 2)} hours')

if epoch != 0:
    print(f'Mean Time per Epoch : {round(training_time / 3600 / epoch, 2)}hours')
#
# ########
# # Eval #
# ########
#
# model.load_state_dict(torch.load('pretrained_AST.pt', weights_only=True))
#
# torch.cuda.synchronize()
# tic = time.time()
#
# model.eval()
#
# count = 0
#
# with torch.no_grad():
#     for real_signals, target_signals, inputs, labels, audio_names in test_loader:
#         inputs = inputs.to(torch.float32)
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#
#         _, predicted = torch.max(outputs.data, 1)
#         predicted_labels = prediction_to_label(predicted)
#
#         estimated_signals = []
#         for i in range(target_signals.size(0)):
#             target_signal = target_signals[i]
#             real_signal = real_signals[i]
#             real_label = labels[i]
#             label = predicted_labels[i]
#             audio_name = audio_names[i]
#
#             if label == '0':
#                 estimated_signal = target_signal
#                 # loss_SISDR = SI_SDR(normSignal_np(estimated_signal.cpu().numpy()), normSignal_np(real_signal.cpu().numpy()))
#             else:
#                 target_signal = target_signal.cpu().numpy()
#                 parameters = compressors[label]
#                 estimated_signal = decompressor(target_signal, 44100, parameters.get('param1'), parameters.get('param2'), parameters.get('param3'),
#                                                           parameters.get('param4'), parameters.get('param5'), parameters.get('param6'), parameters.get('param7'))
#
#             sf.write(output_folder + "/" + audio_name, estimated_signal, 44100)
#
#             count += 1
#             print(count)
#
#         torch.cuda.empty_cache()
