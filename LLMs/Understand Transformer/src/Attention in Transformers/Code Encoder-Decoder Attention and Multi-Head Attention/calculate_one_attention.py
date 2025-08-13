import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor

class Attention(nn.Module):
    def __init__(self, d_model, row_dim=0, col_dim=1):
        super().__init__()

        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query matrix
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key matrix
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value matrix

    def forward(self, encodings_for_q: Tensor, encodings_for_k: Tensor, encodings_for_v: Tensor):
        Q = self.W_q(encodings_for_q)
        K = self.W_k(encodings_for_k)
        V = self.W_v(encodings_for_v)

        # Compute similarity scores based on query and key by using inner product
        # Compute similarities scores: (Q * K^T)
        sims = torch.matmul(Q, K.transpose(self.col_dim, self.row_dim))

        # Scale similarity scores
        scaled_sims = sims / torch.tensor(self.d_model ** 0.5)

        # Apply softmax to get attention weights
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        # Compute attention scores
        attention_scores = torch.matmul(attention_percents, V)
        return attention_scores
    
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
    attention = Attention(d_model)

    # Compute self-attention
    attention_scores = attention(encodings_for_q, encodings_for_k, encodings_for_v)
    print("Shape:", attention_scores.shape)  # Should be (3, 3, 2)
    print("Attention Scores:", attention_scores)