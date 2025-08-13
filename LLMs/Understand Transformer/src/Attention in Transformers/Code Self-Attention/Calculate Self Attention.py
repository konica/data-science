import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, row_dim = 0, col_dim = 1):
        super().__init__()
        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

        # Define weighted matrices for query, key, and value using linear layers
        # These matrices will be learned and updated during training
        self.W_q = nn.Linear(d_model, d_model, bias=False) # exclude bias
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, token_encodings):
        # Compute query, key, and value matrices
        Q = self.W_q(token_encodings) # (3x2)
        K = self.W_k(token_encodings) # (3x2)
        V = self.W_v(token_encodings) # (3x2)

        # Compute similarity scores based on query and key by using inner product
        # Compute similarities scores: (Q * K^T) 
        # Shape of sims will be (num_token_len, num_token_len) (3x3)
        sims = Q @ K.transpose(self.col_dim, self.row_dim) # (3x2)@(2x3) = (3x3)

        # Scale similarity score by dividing sqrt(d_model)
        scaled_sims = sims / self.d_model ** 0.5 # (3x3)

        ## Apply softmax to determine what percent of each token (row)'s value to
        ## use in the final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim) # (3x3)

        # Compute attention scores
        # Shape of attention_scores will be (num_token_len, d_model) (3x2)
        attention_scores = attention_percents @ V # (3x3)@(3x2) = (3x2)
        return attention_scores

if __name__ == "__main__":
    # Example usage
    # Prompt: Write a poem
    # Given its encoding computed below

    token_encodings = torch.tensor(
        [[1.16, 0.23],
         [0.57, 1.36],
         [4.41, -2.16]], 
        dtype=torch.float32)

    torch.manual_seed(42)

    d_model = 2
    self_attention = SelfAttention(d_model)

    # Compute self-attention
    attention_scores = self_attention(token_encodings)
    print("Shape:", attention_scores.shape)  # Should be (3, 3, 2)
    print("Attention Scores:", attention_scores)
