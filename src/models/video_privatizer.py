import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

class VideoPrivatizer(pl.LightningModule):
    def __init__(self, channels):
        super(VideoPrivatizer, self).__init__()

        # Define 3D convolutional layers with the same number of input and output channels
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Apply 3D dropout for regularization
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        # Pass input through the first convolutional layer followed by ReLU and dropout
        y = F.relu(self.conv1(x))
        y = self.dropout(y)

        # Repeat for subsequent layers
        y = F.relu(self.conv2(y))
        y = self.dropout(y)
        y = F.relu(self.conv3(y))
        y = self.dropout(y)
        y = F.relu(self.conv4(y))
        y = self.dropout(y)

        # Add the input (skip connection) to the output to preserve information
        return x + y, y

    def _common_step(self, batch, step_type):
        # Unpack the batch (input video and labels)
        x, _ = batch

        # Forward pass to get output and intermediate features
        output, intermediate = self(x)

        # Log the mean of the output for debugging and monitoring purposes
        #self.log(f"{step_type}_output_mean", output.mean(), prog_bar=True, on_step=True, on_epoch=True)
        return {'output': output, 'intermediate': intermediate}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        # Use Adam optimizer with a learning rate of 1e-3
        return Adam(self.parameters(), lr=1e-3)

    def on_train_epoch_end(self):
        print("\n** on_train_epoch_end **")

    def on_validation_epoch_end(self):
        print("\n** on_validation_epoch_end **")

    def on_test_epoch_end(self):
        print("\n** on_test_epoch_end **")

if __name__ == "__main__":
    # Define input shape (batch size, channels, depth, height, width)
    input_shape = (5, 3, 20, 256, 256)

    # Initialize the VideoPrivatizer model
    model = VideoPrivatizer(channels=input_shape[1])

    # Generate random input data to simulate a batch of videos
    video_input = torch.randn(input_shape)
    labels = torch.zeros(input_shape[0])  # Placeholder labels (not used in this example)
    batch = (video_input, labels)

    # Perform a forward pass and print the output shape
    output, intermediate = model(video_input)
    print(f"Output shape: {output.shape}")
    print(f"Intermediate features shape: {intermediate.shape}")

    # Test the training step
    train_step = model.training_step(batch, batch_idx=0)
    print(f"Training step output mean: {train_step['output'].mean().item()}")

    # Test the validation step
    val_step = model.validation_step(batch, batch_idx=0)
    print(f"Validation step output mean: {val_step['output'].mean().item()}")

    # Test the test step
    test_step = model.test_step(batch, batch_idx=0)
    print(f"Test step output mean: {test_step['output'].mean().item()}")
