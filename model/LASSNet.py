import torch
import torch.nn as nn
import torch.nn.functional as F
from model.text_encoder import Text_Encoder
from model.denseunet import DenseUNet30

class LASSNet(nn.Module):
    def __init__(self, device='cuda'):
        super(LASSNet, self).__init__()
        self.device = device
        self.text_embedder = Text_Encoder(device)
        self.UNet = DenseUNet30(target_sources_num=1, target_sources=['target'], output_channels=1)

    def forward(self, x, caption):
        # x: (Batch, 1, T, F)
        input_ids, attns_mask = self.text_embedder.tokenize(caption)
        cond_vec = self.text_embedder(input_ids, attns_mask)[0]  # (B, D)
        dec_cond_vec = cond_vec

        out = self.UNet(x, cond_vec, dec_cond_vec)  # (B, 3, T, F)
        out = out[:, :, :x.shape[2], :x.shape[3]]

        return out  # (B, 3, T, F)

    def get_tokenizer(self):
        return self.text_embedder.tokenizer