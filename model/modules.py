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
from model.modules import DenseUNet30

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
