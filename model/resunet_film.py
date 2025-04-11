import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.denseunet import DenseUNet30  # thay UNetRes_FiLM bằng DenseUNet30
from torchlibrosa.stft import STFT, ISTFT, magphase

class DenseUNet_Interface(nn.Module):
    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DenseUNet30(target_sources_num=1, target_sources=['target'], output_channels=1).to(self.device)

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)

        self.model.eval()

        self.stft = STFT(n_fft=1024, hop_length=256, win_length=1024).to(self.device)
        self.istft = ISTFT(n_fft=1024, hop_length=256, win_length=1024).to(self.device)

    def forward(self, input_dict):
        waveform = input_dict['mixture'].to(self.device)
        condition = input_dict['condition'].to(self.device)

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, T)

        mag, cos, sin = self.wav_to_spectrogram_phase(waveform.squeeze(1))  # (B, T, F)
        x = mag.unsqueeze(1)  # (B, 1, T, F)

        with torch.no_grad():
            out = self.model(x, condition, condition)  # (B, 3, T, F)
            out = out[:, :, :mag.shape[1], :]
            waveform_out = self.feature_maps_to_wav(out, mag, sin, cos, waveform.shape[2])

        return {'waveform': waveform_out}

    def wav_to_spectrogram_phase(self, input):
        real, imag = self.stft(input)
        mag = (real ** 2 + imag ** 2) ** 0.5
        cos = real / (mag + 1e-8)
        sin = imag / (mag + 1e-8)
        return mag, cos, sin

    def feature_maps_to_wav(self, x, mag, sin_in, cos_in, audio_length):
        B, _, T, F = x.shape
        x = x.reshape(B, 1, 3, T, F)
        mask_mag = torch.sigmoid(x[:, :, 0])
        mask_real = torch.tanh(x[:, :, 1])
        mask_imag = torch.tanh(x[:, :, 2])

        _, mask_cos, mask_sin = magphase(mask_real, mask_imag)

        mag = mag[:, None, :T, :F]
        cos_in = cos_in[:, None, :T, :F]
        sin_in = sin_in[:, None, :T, :F]

        out_cos = cos_in * mask_cos - sin_in * mask_sin
        out_sin = sin_in * mask_cos + cos_in * mask_sin
        out_mag = F.relu_(mag * mask_mag)

        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin

        out_real = out_real.reshape(B, 1, T, F)
        out_imag = out_imag.reshape(B, 1, T, F)

        waveform = self.istft(out_real, out_imag, audio_length)
        waveform = waveform.reshape(B, 1, audio_length)
        return waveform


if __name__ == "__main__":
    model = DenseUNet_Interface(checkpoint_path=None)
    dummy_input = {
        'mixture': torch.randn(1, 160000),
        'condition': torch.randn(1, 512)
    }
    output = model(dummy_input)
    print(output['waveform'].shape)
