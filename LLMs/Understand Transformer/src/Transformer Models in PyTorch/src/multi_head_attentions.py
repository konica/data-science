import torch
import torch.nn as nn
import torch.nn.functional as F
from input_embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model) # concatenate and project head outputs
    
    def split_heads(self, x, batch_size):        
        seq_length = x.size(1)
        # Split the input embeddings and permute        
        print(f"Before split heads shape: {x.shape}")  # Debugging line to check shape after splitting heads
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        print(f"After split heads shape: {x.shape}")  # Debugging line to check shape after splitting heads
        permuted = x.permute(0, 2, 1, 3)
        print(f"Permuted shape: {permuted.shape}")  # Debugging line to check shape after permuting heads
        return permuted

    def compute_attention(self, query, key, value, mask=None):
        # Transpose (-2, -1) affect to two last dimension, ie. (num of elements in embeddings) and seq_length (num of tokens)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def combine_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.split_heads(self.query_linear(query), batch_size)        
        key = self.split_heads(self.key_linear(key), batch_size)        
        value = self.split_heads(self.value_linear(value), batch_size)        
        
        attention_weights = self.compute_attention(query, key, value, mask)        
        output = self.combine_heads(attention_weights, batch_size)

        return self.output_linear(output) ## concatenate and project head outputs
    
if __name__ == "__main__":
    encodings_query = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    embedding_layer = InputEmbeddings(vocab_size=10_000, d_model=512) # 10,000 words in vocab, 512-dimensional embeddings
    embedded_output = embedding_layer(encodings_query)
    print("embedded_output.shape:", embedded_output.shape)  # torch.Size([2, 4, 512])

    positional_encoding = PositionalEncoding(d_model=512, max_seq_length=100)
    output = positional_encoding(embedded_output)
    print("output.shape:", output.shape)  # Should be (2, 4, 512)

    multi_head_attention = MultiHeadAttention(d_model=512, num_heads=8)
    attention_output = multi_head_attention(output, output, output)
    print("attention_output.shape:", attention_output.shape)  # Should be (2, 4, 512)