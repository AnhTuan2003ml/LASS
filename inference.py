import os
import argparse
import torch
import torch.nn as nn
from model.LASSNet import LASSNet
from utils.stft import STFT
from utils.wav_io import load_wav, save_wav
import warnings
warnings.filterwarnings('ignore')

# language descriptions of target sources in the example mixtures
example_captions = {
    'AudioCaps': [
        'a person shouts nearby and then emergency vehicle sirens sounds',
        'a motor vibrates and then revs up and down',
        'people laugh followed by people singing while music plays',
        'someone is typing on a keyboard',
        'distant claps of thunder with rain falling and a man speaking',
        'heavy wind and birds chirping',
        'applauding followed by people singing and a tambourine',
        'a woman is giving a speech',
        'church bells ringing',
        'an adult male is laughing'],
    
    'Human': [
        'a man is speaking with ambulance and police siren sound in the background',
        'the engine sound of a vehicle',
        'a music show is presenting to the public',
        'the sound of hitting the keyboard',
        'very rainy and a man is talking dirty words in the background',
        'a bird is chirping under the thunder storm',
        'a show start with audience applausing and then singing',
        'a female is speaking with clearing throat sounds',
        'someone is striking a large church bell',
        'a man is laughing really hard'
    ]
}

def inference(ckpt_path, query_src, file_path):
    device = 'cuda'
    stft = STFT()
    model = nn.DataParallel(LASSNet(device)).to(device)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    waveform = load_wav(file_path)
    waveform = torch.tensor(waveform).transpose(1,0)
    mixed_mag, mixed_phase = stft.transform(waveform)
    text_query = ['[CLS] ' + example_captions[query_src][0]]  # Only use the first caption
    print(f'Separate target source from {file_path} with text query: "{text_query[0]}"')
    mixed_mag = mixed_mag.transpose(2,1).unsqueeze(0).to(device)
    est_mask = model(mixed_mag, text_query)
    est_mag = est_mask * mixed_mag  
    est_mag = est_mag.squeeze(1)  
    est_mag = est_mag.permute(0, 2, 1) 
    est_wav = stft.inverse(est_mag.cpu().detach(), mixed_phase)
    est_wav = est_wav.squeeze(0).squeeze(0).numpy()  
    
    est_path = f'output/est_single.wav'
    save_wav(est_wav, est_path)
    print(f'Separation done, saving to {est_path} ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='ckpt/LASSNet.pt', help="Checkpoint of pre-trained LASS-Net.")
    parser.add_argument('-q', '--query', type=str, default='AudioCaps', help="Source of text queries, 'AudioCaps' or 'Human'.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the input audio file (WAV).")
    args = parser.parse_args()
    
    os.makedirs('output', exist_ok=True)
    inference(args.checkpoint, args.query, args.file)
