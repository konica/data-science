import torch
import torch.nn as nn

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff) # d_ff is intermediate dimension between two linear layers
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


if __name__ == "__main__":
    # Example usage
    d_model = 512
    d_ff = 2048
    ff_layer = FeedForwardSubLayer(d_model, d_ff)
    sample_attention_output = torch.randn(2, 4, d_model)  # (batch_size, seq_length, d_model)
    output = ff_layer(sample_attention_output)
    print(output.shape)  # Should be (2, 4, d_model)