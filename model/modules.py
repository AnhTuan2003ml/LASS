import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchlibrosa.stft import STFT, ISTFT, magphase

# ======== Các hàm khởi tạo từ base.py ========
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)

def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

def act(x, activation):
    if activation == "relu":
        return F.relu_(x)
    elif activation == "leaky_relu":
        return F.leaky_relu_(x, negative_slope=0.01)
    elif activation == "swish":
        return x * torch.sigmoid(x)
    else:
        raise Exception("Incorrect activation!")

# ======== Base class ========
class Base:
    def __init__(self):
        pass

    def spectrogram(self, input, eps=0.):
        (real, imag) = self.stft(input)
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5

    def spectrogram_phase(self, input, eps=0.):
        (real, imag) = self.stft(input)
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-10):
        sp_list, cos_list, sin_list = [], [], []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)
        return torch.cat(sp_list, dim=1), torch.cat(cos_list, dim=1), torch.cat(sin_list, dim=1)

    def wav_to_spectrogram(self, input, eps=0.):
        sp_list = [self.spectrogram(input[:, channel, :], eps=eps) for channel in range(input.shape[1])]
        return torch.cat(sp_list, dim=1)

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        wav_list = []
        for channel in range(input.shape[1]):
            real, imag = self.stft(input[:, channel, :])
            _, cos, sin = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel:channel+1, :, :] * cos,
                                       spectrogram[:, channel:channel+1, :, :] * sin, length))
        return torch.stack(wav_list, dim=1)

# ======== DenseUNet wrapper model ========

class DenseUNet30(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super(DenseUNet30, self).__init__()
        # Các thông số giống như trong DenseUNet_FiLM, nhưng có thể thêm hoặc bớt tùy theo yêu cầu
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

        # Định nghĩa các tầng như trong DenseUNet_FiLM
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


class DenseUNet_FiLM(nn.Module):
    def __init__(self, input_channels=2, output_channels=1, cond_embedding_dim=10):
        super(DenseUNet_FiLM, self).__init__()

        self.model = DenseUNet30(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_size=cond_embedding_dim
        )

    def forward(self, input_dict):
        """
        Args:
          input_dict: {'mixture': waveform, 'condition': cond_vec}
            - mixture: (B, C, T) - raw waveform
            - condition: (B, D)

        Return:
          dict: {'waveform': separated_waveform}  # (B, C, T)
        """
        return self.model(input_dict)


if __name__ == '__main__':
    model = DenseUNet_FiLM(input_channels=2, output_channels=1, cond_embedding_dim=10)
    cond_vec = torch.randn((1, 10))
    waveform = torch.randn((1, 2, 16000))  # giả lập 1s audio ở 16kHz
    out = model({'mixture': waveform, 'condition': cond_vec})
    print(out['waveform'].shape)  # Kỳ vọng: (1, 1, T)
