import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam

# ConvBackbone che estrae caratteristiche dai dati video
class ConvBackbone(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 256, 256)):
        super(ConvBackbone, self).__init__()
        
        self.input_channels = input_shape[1]

        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=self.input_channels, out_channels=16, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layers
        self.maxpool1 = nn.MaxPool3d(kernel_size=7, stride=2, padding=3)
        self.maxpool2 = nn.MaxPool3d(kernel_size=5, stride=2, padding=2)
        self.maxpool3 = nn.MaxPool3d(kernel_size=5, stride=2, padding=2)
        self.maxpool4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Flatten layer and dropout
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout3d(p=0.1)

        # Calcolo della dimensione finale dell'output
        self._calculate_flatten_size(input_shape)

    def _calculate_flatten_size(self, input_shape):
        dummy_input = torch.zeros(*input_shape)
        x = self.conv1(dummy_input)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        self.flattened_size = self.flatten(x).size(1)

    def forward(self, video_input):
        x_video = self.conv1(video_input)
        x_video = self.maxpool1(x_video)
        x_video = self.conv2(x_video)
        x_video = self.maxpool2(x_video)
        x_video = self.conv3(x_video)
        x_video = self.maxpool3(x_video)
        x_video = self.conv4(x_video)
        x_video = self.maxpool4(x_video)

        x_video = self.flatten(x_video)
        return x_video

# Funzione di test per ConvBackbone
def test_privacy_conv_backbone():
    print("Test: PrivacyConvBackbone")
    input_shape = (8, 3, 20, 256, 256)
    model = ConvBackbone(input_shape)
    video_input = torch.randn(input_shape)
    output = model(video_input)
    print("Output shape:", output.shape)  # Deve essere (batch_size, flattened_size)
    assert output.shape == (8, model.flattened_size), "Errore nel ConvBackbone!"
    print("\nTest ConvBackbone completato con successo!")

# MLP per ciascun metadato (skin color, gender, age)
class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        
        # Layer MLP
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout_fc = nn.Dropout(p=0.4)
        self.output = nn.Linear(256, output_dim)
        self.activation = nn.GELU()

    def forward(self, x_combined):
        x = self.activation(self.fc1(x_combined))
        x = self.dropout_fc(x)
        output = self.output(x)
        return output
    
# Funzione di test per MLP
def test_privacy_mlp():
    print("Test: PrivacyMLP")

    # Parametri di input per ciascun metadato (output del ConvBackbone)
    input_dim = 256  # La dimensione dell'output del ConvBackbone (flattened size)

    # Numero di classi per ciascun metadato
    output_dim_skin_color = 4  # Numero di classi per skin color
    output_dim_gender = 1  # Numero di classi per gender (regressione logistica)
    output_dim_age = 1  # Numero di classi per age (regressione)

    # Modelli MLP per ciascun metadato
    model_skin_color = MLP(input_dim, output_dim_skin_color)
    model_gender = MLP(input_dim, output_dim_gender)
    model_age = MLP(input_dim, output_dim_age)

    # Creiamo l'input (output del ConvBackbone)
    x_skin_color = torch.randn(8, input_dim)  # Output del ConvBackbone
    output_skin_color = model_skin_color(x_skin_color)
    print(f"Skin color output shape: {output_skin_color.shape}")  # Dovrebbe essere (batch_size, 4)

    x_gender = torch.randn(8, input_dim)  # Output del ConvBackbone
    output_gender = model_gender(x_gender)
    print(f"Gender output shape: {output_gender.shape}")  # Dovrebbe essere (batch_size, 1)

    x_age = torch.randn(8, input_dim)  # Output del ConvBackbone
    output_age = model_age(x_age)
    print(f"Age output shape: {output_age.shape}")  # Dovrebbe essere (batch_size, 1)

    print("\nTest MLP completato con successo!")

# PrivacyClassifier che unisce il backbone convoluzionale con MLP per ciascun task
class PrivacyClassifier(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 256, 256), output_dim=(4, 1, 1)):
        super(PrivacyClassifier, self).__init__()
        
        # Backbone convoluzionale
        self.conv_backbone = ConvBackbone(input_shape=input_shape)
        
        # MLP per ogni tipo di metadato
        self.mlp_skin_color = MLP(input_dim=self.conv_backbone.flattened_size, output_dim=output_dim[0])
        self.mlp_gender = MLP(input_dim=self.conv_backbone.flattened_size, output_dim=output_dim[1])
        self.mlp_age = MLP(input_dim=self.conv_backbone.flattened_size, output_dim=output_dim[2])

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
        loss_skin_color = F.cross_entropy(pred_skin_color, labels['skin_color'])
        loss_gender = F.binary_cross_entropy_with_logits(pred_gender, labels['gender'].unsqueeze(1))
        loss_age = F.mse_loss(pred_age, labels['age'].float().unsqueeze(1))

        # Somma delle loss
        total_loss = loss_skin_color + loss_gender + loss_age

        # Logging delle metriche
        self.log(f"{step_type}_loss_skin_color", loss_skin_color, prog_bar=True)
        self.log(f"{step_type}_loss_gender", loss_gender, prog_bar=True)
        self.log(f"{step_type}_loss_age", loss_age, prog_bar=True)
        self.log(f"{step_type}_loss", total_loss, prog_bar=True)

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

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)

    def validation_epoch_end(self, outputs):
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

# Funzione di test per PrivacyClassifier
def test_privacy_classifier():
    print("Test: PrivacyClassifier")
    
    # Configurazione
    input_shape = (8, 3, 20, 256, 256)  # Batch size di 8
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
    print("\n**Testing forward pass**")
    output_skin_color, output_gender, output_age = model(video_input)
    print(f"Skin color output shape: {output_skin_color.shape}")
    print(f"Gender output shape: {output_gender.shape}")
    print(f"Age output shape: {output_age.shape}")

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

if __name__ == "__main__":
    #test_privacy_conv_backbone()
    #test_privacy_mlp()
    test_privacy_classifier()
