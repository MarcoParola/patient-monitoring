import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class CNN3DLightning(pl.LightningModule):
    def __init__(self, in_channels=3, learning_rate=1e-4, kth=False):
        super(CNN3DLightning, self).__init__()
        self.learning_rate = learning_rate
        
         # Initialize the feature output dimension and dropout rate
        
        self.feature_dim = 256
        self.dropout = nn.Dropout3d(p=0.1)
        
        # Block 1
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=3, padding=3)
        self.pool1 = nn.MaxPool3d(kernel_size=7, stride=3, padding=3)
        
        # Block 2
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool3d(kernel_size=5, stride=2, padding=2)
        
        # Block 3
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool3 = nn.MaxPool3d(kernel_size=5, stride=2, padding=2)
        
        # Block 4
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        
        # Flatten the output and extract features
        self.flatten = nn.Flatten()
       
        if kth:
            self.feature_extractor = nn.Linear(128, self.feature_dim)
        else:
            self.feature_extractor = nn.Linear(256, self.feature_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.feature_extractor(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        features = self.forward(x)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
if __name__ == "__main__":
    print("Test: CNN3DLightning")
    
    # Configurazione
    input_channels = 3
    model = CNN3DLightning(in_channels=input_channels)
    
    # Test forward
    x = torch.randn(3, input_channels, 20, 256, 256)
    features = model(x)
    print(f"Features shape: {features.shape}")

    print("\nTest completato con successo!")