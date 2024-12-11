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
        # Passaggio forward
        x_video = self.conv_backbone(video_input)
        output = self.mlp(x_video)
        return output

    def _common_step(self, batch, step_type):
        video_input, labels = batch

        # Predizione
        pred = self(video_input)

        # Calcolo della loss
        loss = F.cross_entropy(pred, labels)

        # Logging delle metriche con WandB tramite self.log
        self.log(f"{step_type}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_accuracy", self.compute_accuracy(pred, labels), prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'logits': pred}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        # Configurazione dell'ottimizzatore
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def on_train_epoch_end(self):
        print("\n** on_train_epoch_end **")

    def on_validation_epoch_end(self):
        print("\n** on_validation_epoch_end **")

    def on_test_epoch_end(self):
        print("\n** on_test_epoch_end **")

    def compute_accuracy(self, preds, labels):
        # Calcolo dell'accuratezza
        _, predicted = torch.max(preds, 1)
        correct = (predicted == labels).float().sum()
        accuracy = correct / labels.size(0)
        return accuracy

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
    output = model(video_input)
    print(f"Output shape: {output.shape}")

    # Test del training_step
    train_loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {train_loss['loss'].item()}")

    # Test del validation_step
    val_loss = model.validation_step(batch, batch_idx=0)
    print(f"Validation loss: {val_loss['loss'].item()}")

    # Test del test_step
    test_loss = model.test_step(batch, batch_idx=0)
    print(f"Test loss: {test_loss['loss'].item()}")

    print("\nTest completato con successo!")
