import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from src.models.DECOMP import decompressor
from src.models.mee_model import MEE
from src.models.tfe_model import TFE, TimeFrequencyCQT_Encoder
import time
import pickle
import yaml
from src.data.AudioDataset import norm_params, get_dataset, CurriculumDataset, AdversarialDataset, create_collate_fn
import auraloss
import soundfile as sf
from sklearn.decomposition import PCA
import pandas as pd
from src.utiles.utile import set_seed, BalancedBatchSampler


def find_real_params(ranges, p):
    m = ranges[:, 0].reshape(1, -1)
    M = ranges[:, 1].reshape(1, -1)
    pp = (M - m) * p + m
    return pp

def normSignal(x):
    x = x - torch.mean(x, dim=2, keepdim=True)
    x = x / torch.sqrt(torch.mean(torch.square(x), dim=2, keepdim=True))
    return x

def loss_params(q, q_hat, x, y, AFX):
    x_hat = AFX(y, q_hat)
    return 10 * MSE(q_hat, q), x_hat

def loss_mel(q, q_hat, x, y, AFX):
    x_hat = AFX(y, q_hat)
    # x = normSignal(x)cf
    # x_hat = normSignal(x_hat)
    return Mel(x_hat, x), x_hat

def get_lossFn(loss_type):
    if loss_type.lower() == 'params':
        return loss_params
    elif loss_type == 'mel':
        return loss_mel
    else:
        raise ValueError("Wrong loss type.")


# Hyperparameters
seed = 1
set_seed(seed)

batch_size = 12
lr = 0.0001
epochs = 200
num_controls = 6
early_stop = True

control_ranges = np.array([[-60, 0],
                           [0, 15],
                           [0, 130],
                           [0, 130],
                           [0, 500],
                           [0, 2000]])
# control_ranges = np.array([[-38, 0],
#                            [0, 7.3],
#                            [0, 5],
#                            [0, 5],
#                            [0, 13],
#                            [0, 705]])

loss_type = "params"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data loading and preprocessing
audio_folder_path = "/home/hsun/Datasets/MedleyDB/30profiles/20dB/all"
train_folder = "/home/hsun/Datasets/MedleyDB/30profiles/20dB/train"
test_folder = "/home/hsun/Datasets/MedleyDB/30profiles/20dB/test"
eval_folder = test_folder
output_folder = "/home/hsun/Datasets/MedleyDB/30profiles/20dB/output_MEE"

compressors = pickle.load(open('../30profiles.pkl', 'rb'))

train_dataset = get_dataset(train_folder, compressors)
test_dataset = get_dataset(test_folder, compressors)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          )
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         )


Encoder = 'MEE'
if Encoder == 'MEE':
    config = yaml.full_load(open('../models/MEE_configs.yaml', 'r'))
    model = MEE(config, num_controls)
elif Encoder == 'TFE':
    model = TFE(samplerate=16000, f_dim=64, t_dim=431, label_dim=num_controls)
    # model = TimeFrequencyCQT_Encoder()
else:
    raise ValueError("Not a correct encoder name !")

model = nn.DataParallel(model)
model = model.to(device)

fft_sizes = [256, 1024, 4096]
hop_sizes = [64, 256, 1024]
win_lengths = [256, 1024, 4096]

MRSTFT = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=fft_sizes,
    hop_sizes=hop_sizes,
    win_lengths=win_lengths,
    w_sc=1,
    device=device
).to(device)
MSE = torch.nn.MSELoss()
Mel = auraloss.freq.MelSTFTLoss(
    sample_rate=22050,
    fft_size=2048,
    hop_size=512,
    win_length=2048,
    n_mels=128,
    device=device,
    w_sc=0
).to(device)

### Training loop with early stopping
train_losses = []
val_losses = []

### Early stopping parameters
min_epoch = 40
min_loss_val = 10
patience = 20  # Number of epochs with no improvement after which training will be stopped
best_val_loss = float('inf')
patience_counter = 0

# criterion = get_lossFn(loss_type)
criterion = MSE
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

torch.cuda.synchronize()
start = time.time()

for epoch in range(epochs):
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    model.train()

    running_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0
    all_labels_train = []
    all_predictions_train = []

    for batch_idx, (inputs, target, labels, real_q, norm_q, names) in enumerate(iter(train_loader)):
        inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
        inputs, norm_q = inputs.unsqueeze(1).to(device), norm_q.to(device)

        q_hat = model(inputs)

        optimizer.zero_grad()
        loss = criterion(norm_q, q_hat)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, target, labels, real_q, norm_q, names) in enumerate(iter(test_loader)):
            inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
            inputs, norm_q = inputs.unsqueeze(1).to(device), norm_q.to(device)

            q_hat = model(inputs)

            loss = criterion(norm_q, q_hat)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f'Epoch {epoch + 1:03d} | Current lr: {learning_rate:.7f} | \n Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    if val_loss <= min_loss_val:
        min_loss_val = val_loss
        best_model = model.state_dict()
        torch.save(best_model, "../../pretrained models/MedleyDB/MEE_MedleyDB_noised_large.pt")
    ## Early stopping check
    if early_stop:
        if epoch < min_epoch:
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

training_time_hours = (end - start) / 3600
time_per_epoch = training_time_hours / (epoch + 1)

best_val_loss = min(val_losses)
best_epoch = val_losses.index(best_val_loss) + 1
print(f'Best validation accuracy: {best_val_loss:.4f} achieved at epoch {best_epoch}')
print(f'Average training time: {time_per_epoch:.4f} h')


########
# eval #
########
eval_losses_MSE = []
eval_losses_Mel = []

model.load_state_dict(torch.load("../../pretrained models/MedleyDB/MEE_MedleyDB_noised_large.pt", weights_only=True))

torch.cuda.synchronize()
tic = time.time()

eval_loss_MSE = 0.0
eval_loss_Mel = 0.0
Estimated_Signals = []
loss_MSE_all = []
loss_Mel_all = []
count = 0

estimated_p = []
real_labels = []

q_tab = {
    'name': [],
    'label': [],
    'real': [],
    'estimated': [],
    'q': []
}

eval_dataset = get_dataset(eval_folder, compressors)
eval_loader = DataLoader(eval_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         )

model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets, labels, real_q, norm_q, names) in enumerate(iter(eval_loader)):
        inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
        inputs, norm_q = inputs.unsqueeze(1).to(device), norm_q.to(device)
        q_hat = model(inputs)
        # q_hat = find_real_params(control_ranges, q_hat.cpu().numpy())
        estimated_p.append(q_hat)
        real_labels.append(labels)

        estimated_signals = []
        for i in range(inputs.size(0)):
            y = inputs[i]
            x_real = targets[i]
            real_label = labels[i]
            real_parameter = real_q[i]
            theta = q_hat[i]
            audio_name = names[i]

            q_tab['name'].append(audio_name)
            q_tab['label'].append(real_label)
            q_tab['real'].append(real_parameter.numpy())
            q_tab['estimated'].append(find_real_params(control_ranges, theta.cpu().numpy()))
            q_tab['q'].append(theta.cpu().numpy())

            if real_label == 'O':
                estimated_signal = x_real
            else:
                y = y.squeeze().cpu().numpy()
                parameters = find_real_params(control_ranges, theta.cpu().numpy())
                parameters = parameters.flatten()
                estimated_signal = decompressor(y, 44100, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2)

            sf.write(output_folder + "/" + audio_name, estimated_signal, 44100)
            # estimated_signals.append(estimated_signal)
            # Estimated_Signals.append(np.array(estimated_signal))

            count += 1
            print(count)

            torch.cuda.empty_cache()

        if count > 2000:
            break

df = pd.DataFrame(q_tab)
df.to_excel("../../results/MedleyDB/q_noised_large.xlsx", index=False)
