import torch
import torch.nn as nn
import math
from input_embeddings import InputEmbeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

if __name__ == "__main__":
    encodings_query = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    embedding_layer = InputEmbeddings(vocab_size=10_000, d_model=512) # 10,000 words in vocab, 512-dimensional embeddings
    embedded_output = embedding_layer(encodings_query)
    print("embedded_output.shape:", embedded_output.shape)  # torch.Size([2, 4, 512])

    positional_encoding = PositionalEncoding(d_model=512, max_seq_length=100)
    output = positional_encoding(embedded_output)
    print("output.shape:", output.shape)  # Should be (2, 4, 512)
