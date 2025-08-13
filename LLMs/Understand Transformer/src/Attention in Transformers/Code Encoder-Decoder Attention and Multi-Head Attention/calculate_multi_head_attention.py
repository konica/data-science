import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor
from calculate_one_attention import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, row_dim=0, col_dim=1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.row_dim = row_dim
        self.col_dim = col_dim

        self.heads = nn.ModuleList([Attention(d_model) for _ in range(num_heads)])

    def forward(self, encodings_for_q: Tensor, encodings_for_k: Tensor, encodings_for_v: Tensor):
        attention_outputs = [head(encodings_for_q, encodings_for_k, encodings_for_v) for head in self.heads]
        return torch.cat(attention_outputs, dim=-1)
    
if __name__ == "__main__":
    # Example usage
    # Prompt: Write a poem
    # Given its encoding computed below

    encodings_for_q = torch.tensor(
        [[1.16, 0.23],
         [0.57, 1.36],
         [4.41, -2.16]], 
        dtype=torch.float32)

    encodings_for_k = torch.tensor(
        [[1.16, 0.23],
         [0.57, 1.36],
         [4.41, -2.16]], 
        dtype=torch.float32)

    encodings_for_v = torch.tensor(
        [[1.16, 0.23],
         [0.57, 1.36],
         [4.41, -2.16]], 
        dtype=torch.float32)

    torch.manual_seed(42)

    d_model = 2
    multi_head_attention = MultiHeadAttention(d_model, num_heads=2)

    # Compute self-attention
    attention_scores = multi_head_attention(encodings_for_q, encodings_for_k, encodings_for_v)
    print("Shape:\n", attention_scores.shape)  # Should be (3, 4)
    print("Attention Scores:\n", attention_scores)