import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from models import Base, init_layer, init_bn
import numpy as np


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate, has_film=False):
        super(DenseLayer, self).__init__()
        self.has_film = has_film
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.init_weights()

    def init_weights(self):
        init_bn(self.norm1)
        init_bn(self.norm2)
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x, film_dict=None):
        out = self.norm1(x)
        if self.has_film and film_dict:
            out += film_dict.get('beta1', 0)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        if self.has_film and film_dict:
            out += film_dict.get('beta2', 0)
        out = F.relu(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate, has_film=False):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.has_film = has_film

        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate, has_film)
            self.layers.append(layer)

    def forward(self, x, film_dict=None):
        for idx, layer in enumerate(self.layers):
            if self.has_film and film_dict and f'layer{idx}' in film_dict:
                x = layer(x, film_dict[f'layer{idx}'])
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()
        self.condition_size = condition_size
        self.modules, _ = self.create_film_modules(film_meta, [])

    def create_film_modules(self, film_meta, ancestor_names):
        modules = {}
        for module_name, value in film_meta.items():
            if isinstance(value, int):
                ancestor_names.append(module_name)
                unique_name = '->'.join(ancestor_names)
                modules[module_name] = self.add_film_layer(value, unique_name)
            elif isinstance(value, dict):
                ancestor_names.append(module_name)
                modules[module_name], _ = self.create_film_modules(value, ancestor_names)
            ancestor_names.pop()
        return modules, ancestor_names

    def add_film_layer(self, num_features, unique_name):
        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(unique_name, layer)
        return layer

    def forward(self, conditions):
        return self.calculate_film_data(conditions, self.modules)

    def calculate_film_data(self, conditions, modules):
        film_data = {}
        for name, module in modules.items():
            if isinstance(module, nn.Module):
                film_data[name] = module(conditions)[:, :, None, None]
            elif isinstance(module, dict):
                film_data[name] = self.calculate_film_data(conditions, module)
        return film_data


class DenseUNet_FiLM(nn.Module, Base):
    def __init__(self, input_channels, output_channels, condition_size):
        super(DenseUNet_FiLM, self).__init__()
        growth_rate = 16
        bn_size = 4
        drop_rate = 0.1
        film = True

        def block(in_ch, layers):
            return DenseBlock(layers, in_ch, growth_rate, bn_size, drop_rate, has_film=film)

        def td(in_ch, out_ch):
            return TransitionDown(in_ch, out_ch)

        def tu(in_ch, out_ch):
            return TransitionUp(in_ch, out_ch)

        self.stft = STFT(1024, 160, 1024, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.istft = ISTFT(1024, 160, 1024, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.K = 3
        self.output_channels = output_channels
        self.target_sources_num = 1

        self.bn0 = nn.BatchNorm2d(513)
        self.pre_conv = nn.Conv2d(input_channels, 32, kernel_size=1)

        # Encoder
        self.encoder_block1 = block(32, 2)
        self.trans_down1 = td(64, 64)

        self.encoder_block2 = block(64, 2)
        self.trans_down2 = td(96, 96)

        self.encoder_block3 = block(96, 2)
        self.trans_down3 = td(128, 128)

        # Bottleneck
        self.bottleneck = block(128, 2)

        # Decoder
        self.trans_up3 = tu(160, 96)
        self.decoder_block3 = block(96 + 128, 2)

        self.trans_up2 = tu(256, 96)
        self.decoder_block2 = block(192, 2)

        self.trans_up1 = tu(224, 64)
        self.decoder_block1 = block(128, 2)

        # Output conv
        self.after_conv = nn.Conv2d(160, output_channels * self.K, kernel_size=1)

        self.film_meta = get_film_meta(self)
        self.film = FiLM(self.film_meta, condition_size)
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']
        film_dict = self.film(conditions)

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag.transpose(1, 3)
        x = self.bn0(x).transpose(1, 3)
        x = x[..., :-1]
        x = self.pre_conv(x)

        x1 = self.encoder_block1(x, film_dict['encoder_block1'])
        x2 = self.encoder_block2(self.trans_down1(x1), film_dict['encoder_block2'])
        x3 = self.encoder_block3(self.trans_down2(x2), film_dict['encoder_block3'])

        x_center = self.bottleneck(self.trans_down3(x3), film_dict['bottleneck'])

        d3 = self.decoder_block3(torch.cat([self.trans_up3(x_center), x3], dim=1), film_dict['decoder_block3'])
        d2 = self.decoder_block2(torch.cat([self.trans_up2(d3), x2], dim=1), film_dict['decoder_block2'])
        up1 = self.trans_up1(d2)
        if up1.shape[2:] != x1.shape[2:]:
            min_h = min(up1.shape[2], x1.shape[2])
            min_w = min(up1.shape[3], x1.shape[3])
            up1 = up1[:, :, :min_h, :min_w]
            x1 = x1[:, :, :min_h, :min_w]

        d1 = self.decoder_block1(torch.cat([up1, x1], dim=1), film_dict['decoder_block1'])

        x = self.after_conv(d1)
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, :mag.shape[2], :]

        separated_audio = self.feature_maps_to_wav(x, mag, sin_in, cos_in, mixtures.shape[2])
        return {'waveform': separated_audio}
