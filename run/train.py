import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import yaml
import time
import auraloss
import pickle

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

from data.AudioDataset import get_2D_dataset, get_dataset
from models.ast_model import ASTModel
from models.mee_model import MEE
from models.tfe_model import TFE
from utiles.utile import set_seed


def normSignal(x):
    x = x - torch.mean(x, dim=1, keepdim=True)
    x = x / torch.sqrt(torch.mean(torch.square(x), dim=1, keepdim=True))
    return x


def normSignal_np(x):
    x = x - np.mean(x)
    x = x / np.sqrt(np.mean(x ** 2))
    return x

class classifier_train:
    def __int__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()

    def _setup(self):
        set_seed(self.config.seed)

        self.model = ASTModel(
            label_dim=self.config.num_classes,
            input_tdim=self.config.target_shape[1],
            input_fdim=self.config.target_shape[0],
            imagenet_pretrain=self.config.imagenet_pretrain,
            audioset_pretrain=self.config.audioset_pretrain
        )

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.compressors = pickle.load(open(self.config.compressor, 'rb'))

        self.compressors_torch = {
            key: {param: torch.tensor(value).to(self.device)
                  for param, value in params.items()}
            for key, params in self.compressors.items()
        }

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_gamma
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _create_dataloaders(self):
        train_dataset = get_2D_dataset(
            self.config.train_folder,
            target_shape=self.config.target_shape
        )
        test_dataset = get_2D_dataset(
            self.config.test_folder,
            target_shape=self.config.target_shape
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for real_signals, target_signals, inputs, labels, audio_name in self.train_loader:
            inputs = inputs.to(torch.float32)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for real_signals, target_signals, inputs, labels, audio_name in self.test_loader:
                inputs = inputs.to(torch.float32)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_loss = val_loss / len(self.test_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def train(self):
        self._create_dataloaders()

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {self.config.n_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            lr = self.optimizer.param_groups[0]['lr']

            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            self.scheduler.step()

            print(f"Epoch {epoch + 1:03d}/{self.config.n_epochs} | "
                  f"LR: {lr:.7f} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience and self.config.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        hours = training_time / 3600
        print(f"Training completed in {hours:.2f} hours")

    def prediction_to_label(predictions):
        p = []
        for pre_label in predictions:
            l = str(pre_label.cpu().numpy())
            p.append(l)
        return p


class regressor_train:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()

    def _setup(self):
        set_seed(self.config.seed)

        if self.config.encoder == 'MEE':
            config_yaml = yaml.full_load(open(self.config.mee_config_path, 'r'))
            self.model = MEE(config_yaml, self.config.num_controls)
        elif self.config.encoder == 'TFE':
            self.model = TFE(
                samplerate=self.config.sample_rate,
                f_dim=self.config.f_dim,
                t_dim=self.config.t_dim,
                label_dim=self.config.num_controls
            )
        else:
            raise ValueError("Invalid encoder name. Use 'MEE' or 'TFE'")

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.control_ranges = np.array(self.config.control_ranges)

        if self.config.loss_type.lower() == 'params':
            self.criterion = torch.nn.MSELoss()
        elif self.config.loss_type.lower() == 'mel':
            self.fft_sizes = self.config.mel_loss.get('fft_sizes', [256, 1024, 4096])
            self.hop_sizes = self.config.mel_loss.get('hop_sizes', [64, 256, 1024])
            self.win_lengths = self.config.mel_loss.get('win_lengths', [256, 1024, 4096])

            self.MRSTFT = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=self.fft_sizes,
                hop_sizes=self.hop_sizes,
                win_lengths=self.win_lengths,
                w_sc=1,
                device=self.device
            ).to(self.device)

            self.Mel = auraloss.freq.MelSTFTLoss(
                sample_rate=self.config.sample_rate,
                fft_size=self.config.mel_loss.get('fft_size', 2048),
                hop_size=self.config.mel_loss.get('hop_size', 512),
                win_length=self.config.mel_loss.get('win_length', 2048),
                n_mels=self.config.mel_loss.get('n_mels', 128),
                device=self.device,
                w_sc=0
            ).to(self.device)
            self.criterion = self._mel_loss
        else:
            raise ValueError("Invalid loss type. Use 'params' or 'mel'")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.gamma
        )

        self.train_losses = []
        self.val_losses = []

    def _mel_loss(self, q_hat, q, x, y, AFX):
        real_q_hat = self.find_real_params(q_hat)
        x_hat = AFX(y, real_q_hat)
        return self.Mel(x_hat, x)

    def find_real_params(self, p):
        m = torch.tensor(self.control_ranges[:, 0], device=self.device).reshape(1, -1)
        M = torch.tensor(self.control_ranges[:, 1], device=self.device).reshape(1, -1)
        pp = (M - m) * p + m
        return pp

    def _create_dataloaders(self):
        compressors = pickle.load(open(self.config.compressors, 'rb'))

        self.train_dataset = get_dataset(
            self.config.train_folder,
            compressors
        )
        self.test_dataset = get_dataset(
            self.config.test_folder,
            compressors
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (inputs, target, labels, real_q, norm_q, names) in enumerate(self.train_loader):
            inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
            inputs = inputs.unsqueeze(1).to(self.device)
            norm_q = norm_q.to(self.device)

            self.optimizer.zero_grad()
            q_hat = self.model(inputs)

            if self.config.loss_type.lower() == 'params':
                loss = self.criterion(norm_q, q_hat)
            else:
                loss = self.criterion(q_hat, norm_q, target, inputs, self.config.AFX)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, target, labels, real_q, norm_q, names) in enumerate(self.test_loader):
                inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
                inputs = inputs.unsqueeze(1).to(self.device)
                norm_q = norm_q.to(self.device)

                q_hat = self.model(inputs)

                if self.config.loss_type.lower() == 'params':
                    loss = self.criterion(norm_q, q_hat)
                else:  # mel损失
                    loss = self.criterion(q_hat, norm_q, target, inputs, self.config.AFX)

                val_loss += loss.item()

        avg_loss = val_loss / len(self.test_loader)
        return avg_loss

    def train(self):
        self._create_dataloaders()

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {self.config.n_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            lr = self.optimizer.param_groups[0]['lr']

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            self.scheduler.step()

            print(f"Epoch {epoch + 1:03d}/{self.config.n_epochs} | "
                  f"LR: {lr:.7f} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch >= self.config.min_epochs and
                    patience_counter >= self.config.patience and
                    self.config.early_stop):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        hours = training_time / 3600

        best_epoch = np.argmin(self.val_losses) + 1
        best_val_loss = self.val_losses[best_epoch - 1]

        print(f"Training completed in {hours:.2f} hours")
        print(f'Best validation loss: {best_val_loss:.4f} achieved at epoch {best_epoch}')
        print(f'Average training time per epoch: {hours / (epoch + 1):.4f} hours')
