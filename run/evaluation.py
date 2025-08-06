import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import auraloss
import pickle
import soundfile as sf
import yaml
import pandas as pd

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

from data.AudioDataset import get_2D_dataset, get_dataset
from models.ast_model import ASTModel
from models.mee_model import MEE
from models.tfe_model import TFE
from models.DECOMP import decompressor
from utiles.utile import set_seed


class classifier_eval:
    def __init__(self, config, input_folder, output_folder):
        self.config = config
        self.input_path = input_folder
        self.output_folder = output_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()

    def _setup(self):
        set_seed(self.config.seed)
        os.makedirs(self.output_folder, exist_ok=True)

        self.model = ASTModel(
            label_dim=self.config.n_class,
            input_tdim=self.config.target_shape.t_dim,
            input_fdim=self.config.target_shape.f_dim,
            imagenet_pretrain=self.config.AST.imagenet_pretrain,
            audioset_pretrain=self.config.AST.audioset_pretrain
        )

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.config.model_save))
        self.model.eval()

        self.compressors = pickle.load(open(self.config.compressors, 'rb'))

        eval_path = os.path.join(self.input_path, "eval")
        self.eval_dataset = get_2D_dataset(
            eval_path,
            t_dim=self.config.target_shape.t_dim,
            f_dim=self.config.target_shape.f_dim,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

    def prediction_to_label(self, predictions):
        return [str(label.cpu().numpy()) for label in predictions]

    def evaluate(self):
        start_time = time.time()
        count = 0

        with torch.no_grad():
            for real_signals, target_signals, inputs, labels, audio_names in self.eval_loader:
                inputs = inputs.to(torch.float32)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted_labels = self.prediction_to_label(predicted)

                for i in range(target_signals.size(0)):
                    target_signal = target_signals[i]
                    real_signal = real_signals[i]
                    label = predicted_labels[i]
                    audio_name = audio_names[i]

                    if label == '0':
                        estimated_signal = target_signal.cpu().numpy()
                    else:
                        target_np = target_signal.cpu().numpy()
                        parameters = self.compressors[label]
                        estimated_signal = decompressor(
                            target_np,
                            self.config.sample_rate,
                            parameters['param1'],
                            parameters['param2'],
                            parameters['param3'],
                            parameters['param4'],
                            parameters['param5'],
                            parameters['param6'],
                            parameters['param7']
                        )

                    output_path = os.path.join(self.output_folder, audio_name)
                    sf.write(output_path, estimated_signal, self.config.sample_rate)

                torch.cuda.empty_cache()

        evaluation_time = time.time() - start_time
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        print(f"Processed {count} samples in total")
        print(f"Output saved to: {self.output_folder}")

class regressor_eval:
    def __init__(self, config, input_folder, output_folder):
        self.config = config
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()

    def _setup(self):
        set_seed(self.config.seed)

        if self.config.encoder == 'MEE':
            config_yaml = yaml.full_load(open("../conf/MEE_configs.yaml", 'r'))
            self.model = MEE(config_yaml, self.config.n_params)
        elif self.config.encoder == 'TFE':
            self.model = TFE(
                samplerate=self.config.sample_rate,
                f_dim=self.config.TFE.f_dim,
                t_dim=self.config.TFE.t_dim,
                label_dim=self.config.n_params
            )
        else:
            raise ValueError("Invalid encoder name. Use 'MEE' or 'TFE'")

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.config.model_save))
        self.model.eval()

        self.control_ranges = np.array(np.array(self.config.control_ranges))

        if self.config.loss_type.lower() == 'params':
            self.criterion = torch.nn.MSELoss()
        elif self.config.loss_type.lower() == 'mel':
            self.fft_sizes = self.config.mel_loss.fft_sizes
            self.hop_sizes = self.config.mel_loss.hop_sizes
            self.win_lengths = self.config.mel_loss.win_lengths

            self.MRSTFT = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=self.fft_sizes,
                hop_sizes=self.hop_sizes,
                win_lengths=self.win_lengths,
                w_sc=1,
                device=self.device
            ).to(self.device)

            self.Mel = auraloss.freq.MelSTFTLoss(
                sample_rate=self.config.sample_rate,
                fft_size=2048,
                hop_size=512,
                win_length=2048,
                n_mels=128,
                device=self.device,
                w_sc=0
            ).to(self.device)
            self.criterion = self._mel_loss
        else:
            raise ValueError("Invalid loss type. Use 'params' or 'mel'")

        compressors = pickle.load(open(self.config.compressors, 'rb'))

        eval_folder = os.path.join(self.input_folder, "eval")
        self.eval_dataset = get_dataset(
            eval_folder,
            compressors
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

    def _mel_loss(self, q_hat, q, x, y, AFX):
        real_q_hat = self.find_real_params(q_hat)
        x_hat = AFX(y, real_q_hat)
        return self.Mel(x_hat, x)

    def find_real_params(self, p):
        m = torch.tensor(self.control_ranges[:, 0], device=self.device).reshape(1, -1)
        M = torch.tensor(self.control_ranges[:, 1], device=self.device).reshape(1, -1)
        pp = (M - m) * p + m
        return pp

    def evaluation(self):
        torch.cuda.synchronize()

        estimated_p = []
        real_labels = []

        q_tab = {
            'name': [],
            'label': [],
            'real': [],
            'estimated': [],
            'q': []
        }

        with torch.no_grad():
            for batch_idx, (inputs, targets, labels, real_q, norm_q, names) in enumerate(iter(self.eval_loader)):
                inputs, norm_q = inputs.to(torch.float32), norm_q.to(torch.float32)
                inputs, norm_q = inputs.unsqueeze(1).to(self.device), norm_q.to(self.device)
                q_hat = self.model(inputs)
                estimated_p.append(q_hat)
                real_labels.append(labels)

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
                    q_tab['estimated'].append(self.find_real_params(theta.cpu().numpy()))
                    q_tab['q'].append(theta.cpu().numpy())

                    if real_label == 'O':
                        estimated_signal = x_real
                    else:
                        y = y.squeeze().cpu().numpy()
                        parameters = self.find_real_params(theta.cpu().numpy())
                        parameters = parameters.flatten()
                        estimated_signal = decompressor(y,
                                                        self.config.sample_rate,
                                                        parameters[0],
                                                        parameters[1],
                                                        parameters[2],
                                                        parameters[3],
                                                        parameters[4],
                                                        parameters[5],
                                                        2)

                    sf.write(self.output_folder + "/" + audio_name, estimated_signal, self.config.sample_rate)

                    torch.cuda.empty_cache()

        df = pd.DataFrame(q_tab)
        df.to_excel("../results/parameters.xlsx", index=False)
