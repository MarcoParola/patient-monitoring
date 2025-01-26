import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        
        # Layer MLP
        self.fc = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        output = self.output(x)
        return output
    
# Test MLP
if __name__ == "__main__":
    input_dim = 256  # From CNN feature extractor
    batch_size = 8
    model = MLP(input_dim=input_dim)

    # Random input tensor simulating feature vectors from CNN
    features = torch.randn(batch_size, input_dim)

    # Forward pass
    logits = model(features)
    print(f"Logits Shape: {logits.shape}")  # Expecting (batch_size, output_dim)


#self.classifier = MLP(input_dim=256, output_dim=9)