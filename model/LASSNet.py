import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import Text_Encoder
from .denseunet_film import DenseUNet30  # Import DenseUNet30 class from your previous code

class LASSNet(nn.Module):
    def __init__(self, device='cuda', condition_size=256, input_channels=1, output_channels=1):
        super(LASSNet, self).__init__()
        # Text embedding
        self.text_embedder = Text_Encoder(device)
        
        # DenseUNet30 model
        self.UNet = DenseUNet30(input_channels=input_channels, 
                                output_channels=output_channels, 
                                condition_size=condition_size)

    def forward(self, x, caption):
        # x: (Batch, 1, T, 128) - Input mixture
        input_ids, attns_mask = self.text_embedder.tokenize(caption)
        
        # Get the conditioning vector from the text encoder
        cond_vec = self.text_embedder(input_ids, attns_mask)[0]
        dec_cond_vec = cond_vec

        # Generate mask using DenseUNet30
        mask = self.UNet({'mixture': x, 'condition': dec_cond_vec})
        mask = torch.sigmoid(mask['waveform'])
        return mask

    def get_tokenizer(self):
        return self.text_embedder.tokenizer
