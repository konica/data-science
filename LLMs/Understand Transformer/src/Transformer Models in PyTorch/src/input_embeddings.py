import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # Scale by sqrt(d_model) is a standard practice
    
if __name__ == "__main__":
    encodings_query = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    embedding_layer = InputEmbeddings(vocab_size=10_000, d_model=512) # 10,000 words in vocab, 512-dimensional embeddings
    embedded_output = embedding_layer(encodings_query)
    print("embedded_output.shape:", embedded_output.shape)  # torch.Size([2, 4, 512])