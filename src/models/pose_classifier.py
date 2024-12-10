import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from src.models.conv_backbone import CNN3DLightning
from src.models.mlp import MLP

class PoseClassifier(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 256, 256), output_dim=9):
        super(PoseClassifier, self).__init__()
        
        # Backbone convoluzionale
        self.conv_backbone = CNN3DLightning(in_channels=input_shape[1])
        
        # MLP per classificazione finale
        self.mlp = MLP(input_dim=self.conv_backbone.feature_dim, output_dim=output_dim)

    def forward(self, video_input):
        # Caratteristiche video (output dal ConvBackbone)
        x_video = self.conv_backbone(video_input)
        
        # Predizioni finali (unico tensore per tutte le classi)
        output = self.mlp(x_video)
        return output
        
    def _common_step(self, batch, step_type):
        video_input, labels = batch
        pred = self(video_input)

        # Calcolo delle loss
        loss = F.cross_entropy(pred, labels)

        # Logging delle metriche
        self.log(f"{step_type}_loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)

    def on_train_start(self):
        self.log("train_start", torch.tensor(1.0))

    def on_train_end(self):
        self.log("train_end", torch.tensor(1.0))

    def on_epoch_start(self):
        self.log("epoch_start", self.current_epoch)

    def on_epoch_end(self):
        self.log("epoch_end", self.current_epoch)

if __name__ == "__main__":
    print("Test: PoseClassifier")
    
    # Configurazione
    input_shape = (5, 3, 20, 256, 256)  # Batch size di 5
    num_classes = 9  # Numero di classi unificate
    model = PoseClassifier(input_shape=input_shape, output_dim=num_classes)
    
    # Generazione di input casuali
    video_input = torch.randn(input_shape)  # Input video
    labels = torch.randint(0, num_classes, (input_shape[0],))  # Classi unificate
    batch = (video_input, labels)

    # Test del metodo forward
    print("\n**Testing forward pass**")
    output = model(video_input)
    print(f"Output shape: {output.shape}")

    # Test del training_step
    print("\n**Testing training_step**")
    train_loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {train_loss.item()}")

    # Test del validation_step
    print("\n**Testing validation_step**")
    val_loss = model.validation_step(batch, batch_idx=0)
    print(f"Validation loss: {val_loss.item()}")

    # Test del test_step
    print("\n**Testing test_step**")
    test_loss = model.test_step(batch, batch_idx=0)
    print(f"Test loss: {test_loss.item()}")

    # Test del configure_optimizers
    print("\n**Testing configure_optimizers**")
    optimizer = model.configure_optimizers()
    print(f"Optimizer: {optimizer}")

    # Test callback methods
    print("\n**Testing callback methods**")
    model.on_train_start()
    model.on_epoch_start()
    model.on_epoch_end()
    model.on_train_end()

    print("\nTest completato con successo!")
