# denseunet_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from model.base import Base, init_layer, init_bn
import numpy as np


# BẬT chế độ benchmark để tăng tốc nếu đầu vào ổn định
torch.backends.cudnn.benchmark = True


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
            layer = DenseLayer(
            in_channels + i * growth_rate, growth_rate, bn_size, drop_rate, has_film
            )
            self.layers.append(layer)

    def forward(self, x, film_dict=None):
        # print("1_DenseBlock: ",x.shape)
        for idx, layer in enumerate(self.layers):
            if self.has_film and film_dict and f'layer{idx}' in film_dict: 
                x = layer(x, film_dict[f'layer{idx}'])
                # print("2_DenseBlock: ",x.shape)
            else:
                # print("3_DenseBlock_ :",x.shape)
                x = layer(x)
                # print("4_DenseBlock: ",x.shape)
        return x



class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.trans_conv(x)


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



def get_film_meta(module):
    film_meta = {}
    if hasattr(module, 'has_film') and module.has_film:
        if isinstance(module, DenseLayer):
            film_meta['beta1'] = module.norm1.num_features
            film_meta['beta2'] = module.norm2.num_features

    for name, child in module.named_children():
        meta = get_film_meta(child)
        if meta:
            # Nếu là DenseBlock thì lưu dạng layer0, layer1, ...
            if isinstance(module, DenseBlock):
                film_meta[name] = meta
            else:
                film_meta[name] = meta

    return film_meta

class DenseUNet30(nn.Module,Base):
    def __init__(self, input_channels, output_channels, condition_size):
        super(DenseUNet30, self).__init__()
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
                # encoder
        self.encoder_block1 = block(32, 2)       # out: 64
        self.trans_down1 = td(64, 64)

        self.encoder_block2 = block(64, 2)       # out: 96
        self.trans_down2 = td(96, 96)

        self.encoder_block3 = block(96, 2)       # out: 128
        self.trans_down3 = td(128, 128)

        # bottleneck
        self.bottleneck = block(128, 2)          # out: 160

        # decoder
        self.trans_up3 = tu(160, 96)             # upsample: 96
        self.decoder_block3 = block(96 + 128, 2) # input = up3 + x3 = 224

        self.trans_up2 = tu(256, 96)
        self.decoder_block2 = block(192, 2)  # = 160

        self.trans_up1 = tu(224, 64)
        self.decoder_block1 = block(128, 2)  # = 96

        # output conv
        self.after_conv = nn.Conv2d(160, output_channels * self.K, kernel_size=1)

        self.film_meta = get_film_meta(self)
        self.film = FiLM(self.film_meta, condition_size)
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        # Reshape input_tensor để tách các mask
        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # K = 3 → 0: mask_mag, 1: mask_real, 2: mask_imag
        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        mask_real = torch.tanh(x[:, :, :, 1, :, :])
        mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(mask_real, mask_imag)

        # Đồng bộ kích thước các đầu vào (time, freq)
        def match_shape(a, ref):
            min_t = min(a.shape[-2], ref.shape[-2])
            min_f = min(a.shape[-1], ref.shape[-1])
            return a[..., :min_t, :min_f]

        sp = match_shape(sp, mask_mag)
        sin_in = match_shape(sin_in, mask_mag)
        cos_in = match_shape(cos_in, mask_mag)
        mask_cos = match_shape(mask_cos, sp)
        mask_sin = match_shape(mask_sin, sp)
        mask_mag = match_shape(mask_mag, sp)

        # Tính out_cos, out_sin theo công thức phase chuẩn
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )

        # Tính magnitude đầu ra
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)

        # Tính real & imag
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin

        # Đưa về dạng (N, 1, T, F) để đưa vào ISTFT
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            out_real.shape[-2],
            out_real.shape[-1],
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT bằng self.istft (torchlibrosa)
        x = self.istft(out_real, out_imag, audio_length)

        # Reshape lại ra (B, C, T)
        waveform = x.reshape(
            batch_size,
            self.target_sources_num * self.output_channels,
            audio_length
        )

        return waveform




    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']
        film_dict = self.film(conditions)

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag.transpose(1, 3)
        # print("2_DenseUNet30_xshape",x.shape)
        x = self.bn0(x).transpose(1, 3)
        # print("3_DenseUNet30_xshape",x.shape)
        x = x[..., :-1]
        # print("4_DenseUNet30_xshape",x.shape)
        x = self.pre_conv(x)
        # print("5_DenseUNet30_xshape: ",x.shape)
        x1 = self.encoder_block1(x, film_dict['encoder_block1'])
        # print("DenseUNet30_x_1shape",x.shape)
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