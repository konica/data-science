import torch
import torch.nn as nn

from transformer_encoder import TransformerEncoder
from transformer_decoder_v2 import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_len, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size,
                                           d_model, num_heads, num_layers,
                                           d_ff, dropout, max_seq_len)
        self.decoder = TransformerDecoder(vocab_size,
                                           d_model, num_heads, num_layers,
                                           d_ff, dropout, max_seq_len)

    def forward(self, x, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(x, encoder_output,
                                      tgt_mask, cross_mask)
        return decoder_output
