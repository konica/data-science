import torch
import torch.nn as nn
from torch.nn import functional as F

from input_embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding
from decoder_layer_v2 import DecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size) # Final linear layer to project to vocabulary size

    def forward(self, x, y, tgt_mask, cross_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, y, tgt_mask, cross_mask)

        # Add self.fc and softmax activation in forward pass
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

if __name__ == "__main__":
    vocab_size = 10_000  # Example vocabulary size
    d_model = 512  # Example model dimension
    num_layers = 6  # Number of decoder layers
    num_heads = 8  # Number of attention heads
    d_ff = 2048  # Dimension of feed-forward network
    dropout = 0.1  # Dropout rate
    max_seq_length = 4  # Example maximum sequence length
    decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length=max_seq_length)
    
    torch.manual_seed(42)  # For reproducibility

    input_sequence = torch.randint(0, vocab_size, (1, max_seq_length))  # Example input sequence
    tgt_mask = torch.tril(torch.ones((max_seq_length, max_seq_length))).bool()  # Example target mask
    output = decoder(input_sequence, tgt_mask)
    print("Input sequence:", input_sequence)  # Should be (batch_size, seq_length)
    print("Output sequence:", output.shape)  # Should be (batch_size, seq_length, vocab_size)

    # Get next word from output
    next_word = output[:, -1, :].argmax(dim=-1)
    print("Next word:", next_word)

    # Get all previous words from output
    all_prev_words = output[:, :-1, :].argmax(dim=-1)
    print("All previous words:", all_prev_words)
