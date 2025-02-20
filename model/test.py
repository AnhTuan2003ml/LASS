import os
import torch
import torch.nn as nn
from model.LASSNet import LASSNet
from utils.stft import STFT
from utils.wav_io import load_wav, save_wav
import warnings
warnings.filterwarnings('ignore')

def inference(ckpt_path, text_query):
    device = 'cpu'
    mixtures_dir = 'examples'
    stft = STFT()
    model = nn.DataParallel(LASSNet(device)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model']
    # Loại bỏ thông số không tương thích
    state_dict = {k: v for k, v in state_dict.items() if k != 'module.text_embedder.bert_layer.embeddings.position_ids'}
    model.load_state_dict(state_dict)
    model.eval()
    wav_path = f'{mixtures_dir}/mix1.wav'
    waveform = load_wav(wav_path)
    waveform = torch.tensor(waveform).transpose(1,0)
    mixed_mag, mixed_phase = stft.transform(waveform)
    mixed_mag = mixed_mag.transpose(2,1).unsqueeze(0).to(device)
    est_mask = model(mixed_mag, text_query)
    est_mag = est_mask * mixed_mag  
    est_mag = est_mag.squeeze(1)  
    est_mag = est_mag.permute(0, 2, 1) 
    est_wav = stft.inverse(est_mag.cpu().detach(), mixed_phase)
    est_wav = est_wav.squeeze(0).squeeze(0).numpy()  
    os.makedirs('output', exist_ok=True)
    est_path = 'output/est1.wav'
    save_wav(est_wav, est_path)
    print(f'Separation done, saving to {est_path} ...')

if __name__ == '__main__':
    ckpt_path = 'ckpt/LASSNet.pt'
    text_query = ['[CLS] a man is speaking and police siren sound in the background']
    inference(ckpt_path, text_query)
