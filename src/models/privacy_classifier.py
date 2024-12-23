import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from src.models.conv_backbone import CNN3DLightning
from src.models.mlp import MLP

# PrivacyClassifier che unisce il backbone convoluzionale con MLP per ciascun task
class PrivacyClassifier(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 256, 256), output_dim=(4, 1, 1)):
        super(PrivacyClassifier, self).__init__()
        
        # Backbone convoluzionale
        self.conv_backbone = CNN3DLightning(in_channels=input_shape[1])
        
        # MLP per ogni tipo di metadato
        self.mlp_skin_color = MLP(input_dim=self.conv_backbone.feature_dim, output_dim=output_dim[0])
        self.mlp_gender = MLP(input_dim=self.conv_backbone.feature_dim, output_dim=output_dim[1])
        self.mlp_age = MLP(input_dim=self.conv_backbone.feature_dim, output_dim=output_dim[2])

    def forward(self, video_input):
        # Caratteristiche video (output dal ConvBackbone)
        x_video = self.conv_backbone(video_input)
        
        # Predizioni senza concatenare i metadati
        output_skin_color = self.mlp_skin_color(x_video)
        output_gender = self.mlp_gender(x_video)
        output_age = self.mlp_age(x_video)

        return output_skin_color, output_gender, output_age

    def _common_step(self, batch, step_type):
        video_input, labels = batch
        pred_skin_color, pred_gender, pred_age = self(video_input)

        # Calcolo delle loss
        loss_skin_color = F.cross_entropy(pred_skin_color, labels[1])
        loss_gender = F.binary_cross_entropy_with_logits(pred_gender, labels[2].unsqueeze(1))
        loss_age = F.mse_loss(pred_age, labels[3].unsqueeze(1).float())

        # Somma delle loss
        total_loss = loss_skin_color + loss_gender + loss_age

        # Calcolo dell'accuracy
        acc_skin_color = self.compute_accuracy_multi_class(pred_skin_color, labels[1])
        acc_gender = self.compute_accuracy_binary(pred_gender, labels[2])

        # Logging delle metriche
        self.log(f"{step_type}_loss_skin_color", loss_skin_color, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_loss_gender", loss_gender, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_loss_age", loss_age, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)

        self.log(f"{step_type}_acc_skin_color", acc_skin_color, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_acc_gender", acc_gender, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def on_train_epoch_end(self):
        print("\n** on_train_epoch_end **")

    def on_validation_epoch_end(self):
        print("\n** on_validation_epoch_end **")

    def on_test_epoch_end(self):
        print("\n** on_test_epoch_end **")

    def compute_accuracy_multi_class(self, preds, labels):
        _, predicted = torch.max(preds, 1)
        correct = (predicted == labels).float().sum()
        return correct / labels.size(0)

    def compute_accuracy_binary(self, preds, labels):
        predicted = (torch.sigmoid(preds) > 0.5).float()
        correct = (predicted == labels).float().sum()
        return correct / labels.size(0)

if __name__ == "__main__":
    print("Test: PrivacyClassifier")
    
    # Configurazione
    input_shape = (5, 3, 20, 256, 256)  # Batch size di 5
    num_classes = (4, 1, 1)  # Numero di classi per skin color, gender e age
    model = PrivacyClassifier(input_shape=input_shape, output_dim=num_classes)
    
    # Generazione di input casuali
    video_input = torch.randn(input_shape)  # Input video
    labels = {
        'skin_color': torch.randint(0, num_classes[0], (input_shape[0],)),        # Classi per skin color
        'gender': torch.randint(0, 2, (input_shape[0],), dtype=torch.float32),    # Valori binari per gender
        'age': torch.randn(input_shape[0])                                        # Valori continui per age
    }
    batch = (video_input, labels)
    
    # Test del metodo forward
    output_skin_color, output_gender, output_age = model(video_input)
    print(f"Output Skin Color shape: {output_skin_color.shape}")
    print(f"Output Gender shape: {output_gender.shape}")
    print(f"Output Age shape: {output_age.shape}")

    # Test del training_step
    train_loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {train_loss.item()}")

    # Test del validation_step
    val_loss = model.validation_step(batch, batch_idx=0)
    print(f"Validation loss: {val_loss.item()}")

    # Test del test_step
    test_loss = model.test_step(batch, batch_idx=0)
    print(f"Test loss: {test_loss.item()}")

    print("\nTest completato con successo!")
