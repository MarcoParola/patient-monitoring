import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VideoPrivatizer(pl.LightningModule):
    def __init__(self, channels):
        super(VideoPrivatizer, self).__init__()
        
        # Layer convoluzionali 3D
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Dropout 3D con probabilità 0.1
        self.dropout = nn.Dropout3d(p=0.1)
        
    def forward(self, x):
        # Passaggio attraverso i layer convoluzionali e applicazione di ReLU e Dropout
        y = F.relu(self.conv1(x))
        y = self.dropout(y)
        y = F.relu(self.conv2(y))
        y = self.dropout(y)
        y = F.relu(self.conv3(y))
        y = self.dropout(y)
        y = F.relu(self.conv4(y))
        y = self.dropout(y)
        
        # Skip connection: somma dell'input con l'output
        return x + y, y  # anche output intermedio

    def _common_step(self, batch, batch_idx):
        x, _ = batch  # Supponiamo che `batch` contenga il video e le etichette (non usate in questo caso)
        output, intermediate = self(x)  # Otteniamo sia l'output finale che quello intermedio
        return output, intermediate

    def training_step(self, batch, batch_idx):
        # Utilizziamo la _common_step per calcolare l'output
        output, intermediate = self._common_step(batch, batch_idx)
        
        # Calcoliamo la loss (ad esempio, MSE per regressione)
        loss = F.mse_loss(output, batch[0])  # Assumiamo che batch[0] contenga l'input
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, intermediate = self._common_step(batch, batch_idx)
        
        # Calcoliamo la loss anche per la validazione
        loss = F.mse_loss(output, batch[0])  # Assumiamo che batch[0] contenga l'input
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        # Configurazione di un ottimizzatore Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Test della rete (senza Trainer e Dataloader)
if __name__ == "__main__":
    # Dimensioni input
    batch_size, channels, depth, height, width = 4, 3, 20, 256, 256
    input_tensor = torch.randn(batch_size, channels, depth, height, width)
    
    # Creazione del modello
    model = VideoPrivatizer(channels)
    
    # Simuliamo un singolo passaggio attraverso la rete
    output_tensor, intermediate_output = model(input_tensor)
    
    # Verifica delle dimensioni
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape (final): {output_tensor.shape}")
    print(f"Output shape (intermediate): {intermediate_output.shape}")
    
    # Calcoliamo la loss per il test
    loss = F.mse_loss(output_tensor, input_tensor)
    print(f"Loss: {loss.item()}")
