import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
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

    
    def forward(self, token_encodings, masked_matrix = None):
        """
        Forward pass for masked self-attention.
        Where masked_matrix is used to apply attention masking. 
        It's a boolean matrix with True for masking out tokens and False for keeping them.
        """
        # Compute query, key, and value matrices
        Q = self.W_q(token_encodings)
        K = self.W_k(token_encodings)
        V = self.W_v(token_encodings)

        # Compute similarity scores based on query and key by using inner product
        # Compute similarities scores: (Q * K^T)
        sims = torch.matmul(Q, K.transpose(self.col_dim, self.row_dim))

        # Scale similarity score by dividing sqrt(d_model)
        scaled_sims = sims / torch.tensor(self.d_model ** 0.5)

        # Apply mask (if provided) to the scaled similarity scores
        if masked_matrix is not None:
            scaled_sims = scaled_sims.masked_fill(mask=masked_matrix, value=torch.tensor(float('-inf')))

        ## Apply softmax to determine what percent of each token (row)'s value to
        ## use in the final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        # Compute attention scores
        attention_scores = attention_percents @ V
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
    
    num_tokens = token_encodings.size(0)

    torch.manual_seed(42)

    d_model = 2
    self_attention = MaskedSelfAttention(d_model)

    # Create a mask (for example, to mask out the second token)
    masked_matrix = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.bool))
    masked_matrix = masked_matrix == 0
    print("Masked Matrix:\n", masked_matrix)

    # Compute self-attention
    attention_scores = self_attention(token_encodings, masked_matrix)
    print("Shape:\n", attention_scores.shape)  # Should be (3, 2)
    print("Masked Self-Attention Scores:\n", attention_scores)
