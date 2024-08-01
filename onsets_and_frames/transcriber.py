"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, labeled=True, pseudo_labels=None, ignore_index=-1, 
                     loss_weights={'onset': 1., 'offset': 1., 'frame': 1., 'velocity': 1.}, mel=None):
        
        audio = batch['audio']

        if labeled:
            onset_label = batch['onset']
            offset_label = batch['offset']
            frame_label = batch['frame']
            velocity_label = batch['velocity']

        elif pseudo_labels is not None:
            onset_label = pseudo_labels['onset']
            offset_label = pseudo_labels['offset']
            frame_label = pseudo_labels['frame']
            velocity_label = pseudo_labels['velocity']

        if mel is None:
            mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        if labeled or (pseudo_labels is not None): 
            predictions = {
                'onset': onset_pred.reshape(*onset_label.shape),
                'offset': offset_pred.reshape(*offset_label.shape),
                'frame': frame_pred.reshape(*frame_label.shape),
                'velocity': velocity_pred.reshape(*velocity_label.shape)
            }

        else:
            predictions = {
                'onset': onset_pred,
                'offset': offset_pred,
                'frame': frame_pred,
                'velocity': velocity_pred
            }

        if labeled:
            losses = {
                'loss/onset': loss_weights['onset'] * F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/offset': loss_weights['offset'] * F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': loss_weights['frame'] * F.binary_cross_entropy(predictions['frame'], frame_label),
                'loss/velocity': loss_weights['velocity'] * self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
        elif pseudo_labels is not None:
            losses = {
                'loss/onset': loss_weights['onset'] * (onset_label != ignore_index).sum() / onset_label.numel() * \
                    F.binary_cross_entropy(predictions['onset'][onset_label != ignore_index], onset_label[onset_label != ignore_index]) \
                        if (onset_label != ignore_index).sum() > 0 else torch.tensor([0]).to(predictions['onset'].device),

                'loss/offset': loss_weights['offset'] * (offset_label != ignore_index).sum() / offset_label.numel() * \
                    F.binary_cross_entropy(predictions['offset'][offset_label != ignore_index], offset_label[offset_label != ignore_index]) \
                        if (offset_label != ignore_index).sum() > 0 else torch.tensor([0]).to(predictions['offset'].device),

                'loss/frame': loss_weights['frame'] * (frame_label != ignore_index).sum() / frame_label.numel() * \
                    F.binary_cross_entropy(predictions['frame'][frame_label != ignore_index], frame_label[frame_label != ignore_index]) \
                        if (frame_label != ignore_index).sum() > 0 else torch.tensor([0]).to(predictions['frame'].device),

                'loss/velocity': loss_weights['velocity'] * self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
        else:
            losses = None

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
