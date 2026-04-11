import torch
import torch.nn as nn


class HubertLabseConcat(nn.Module):
    def __init__(self, in_features_text, in_features_image, mode="joint"):
        super(HubertLabseConcat, self).__init__()
        self.mode = mode
        self.image_seq = nn.Sequential(
            nn.Linear(in_features_image, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(768, 576),
            nn.BatchNorm1d(576),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(576, 768),
        )
        self.audio_seq = nn.Sequential(
            nn.Linear(in_features_text, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(768, 576),
            nn.BatchNorm1d(576),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(576, 768),
        )

        self.mix_seq = nn.Sequential(
            nn.Linear(2 * 768, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 768),
        )

    def forward(self, x_audio, x_image):
        x1 = self.audio_seq(x_audio)
        if self.mode == "audio":
            return x1
        x2 = self.image_seq(x_image)
        if self.mode == "image":
            return x2
        concats = torch.cat((x1, x2), dim=1)
        x = self.mix_seq(concats)
        return x


class Wav2vecConcat(nn.Module):
    def __init__(self, in_features_text, in_features_image):
        super(Wav2vecConcat, self).__init__()
        self.image_seq = nn.Sequential(
            nn.Linear(in_features_image, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(768, 576),
            nn.BatchNorm1d(576),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(576, 768),
        )
        self.audio_seq = nn.Sequential(
            nn.Linear(in_features_text, 700),
            nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(700, 576),
            nn.BatchNorm1d(576),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(576, 768),
        )

        self.mix_seq = nn.Sequential(
            nn.Linear(2 * in_features_text, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 800),
            nn.LeakyReLU(),
            nn.Linear(800, in_features_text),
        )

    def forward(self, x_audio, x_image):
        x1 = self.audio_seq(x_audio)
        x2 = self.image_seq(x_image)
        concats = torch.cat((x1, x2), dim=1)
        x = self.mix_seq(concats)
        return x


class HubertLabseGating(nn.Module):
    def __init__(self, in_features_text, in_features_image):
        super(HubertLabseGating, self).__init__()
        self.image_seq = nn.Sequential(
            nn.Linear(in_features_image, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(768, 576),
            nn.BatchNorm1d(576),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(576, 768),
        )
        self.text_seq = nn.Sequential(
            nn.Linear(in_features_text, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(768, 576),
            nn.BatchNorm1d(576),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(576, 768),
        )

        self.input_gate_text = nn.Sequential(nn.Linear(768, 768), nn.Sigmoid())
        self.input_gate_image = nn.Sequential(nn.Linear(768, 768), nn.Sigmoid())

        self.forget_gate_text = nn.Sequential(nn.Linear(768, 768), nn.Sigmoid())
        self.forget_gate_image = nn.Sequential(nn.Linear(768, 768), nn.Sigmoid())

        self.tanh_text = nn.Sequential(nn.Linear(768, 768), nn.Tanh())
        self.tanh_image = nn.Sequential(nn.Linear(768, 768), nn.Tanh())

        self.weight = nn.Sequential(nn.Linear(768 * 2, 768 * 2))

    def forward(self, x_text, x_image):
        x1 = self.text_seq(x_text)
        x2 = self.image_seq(x_image)

        input_gate_text = self.input_gate_text(x1)
        input_gate_image = self.input_gate_image(x2)

        forget_gate_text = self.forget_gate_text(x1)
        forget_gate_image = self.forget_gate_image(x2)

        new_cell_state_text = self.tanh_text(x1)
        new_cell_state_image = self.tanh_image(x2)

        x1 = input_gate_text * new_cell_state_text + forget_gate_text * x1
        x2 = input_gate_image * new_cell_state_image + forget_gate_image * x2

        weight = torch.softmax(self.weight(torch.cat((x1, x2), dim=-1)), dim=-1)
        x = weight[:, :768] * x1 + weight[:, 768:] * x2
        return x

