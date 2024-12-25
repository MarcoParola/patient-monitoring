import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

# Define a neural network model for video privacy preservation
class VideoPrivatizer(pl.LightningModule):
    def __init__(self, channels, learning_rate):
        """
        A neural network model for video privacy preservation using 3D convolutions.
        Args:
            channels (int): Number of channels in the input video.
            learning_rate (float): The learning rate for the optimizer.
        """
        super(VideoPrivatizer, self).__init__()

        # Set the learning rate for the optimizer
        self.learning_rate = learning_rate

        # Define 3D convolutional layers with equal input and output channels
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Apply 3D dropout for regularization
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        Returns:
            Tuple[Tensor, Tensor]: Final output and intermediate features.
        """
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
        """
        Shared logic for training, validation, and test steps.
        Args:
            batch (tuple): Tuple containing input video and placeholder labels.
            step_type (str): Type of step ('train', 'val', or 'test').
        Returns:
            dict: Dictionary with loss, output, and intermediate features.
        """
        x, _ = batch

        # Forward pass to get output and intermediate features
        output, intermediate = self(x)

        # Calculate loss as the mean squared error between input and output
        loss = F.mse_loss(output, x)

        # Log the loss value
        self.log(f"{step_type}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'output': output, 'intermediate': intermediate}

    def training_step(self, batch, batch_idx):
        """
        Training step logic.
        Args:
            batch (tuple): Training batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Output of _common_step for the training step.
        """
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step logic.
        Args:
            batch (tuple): Validation batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Output of _common_step for the validation step.
        """
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        """
        Test step logic.
        Args:
            batch (tuple): Test batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Output of _common_step for the test step.
        """
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        """
        Configure the optimizer for the model using the learning rate passed in the constructor.
        Returns:
            Optimizer: Adam optimizer with the specified learning rate.
        """
        return Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        """Hook executed at the end of each training epoch."""
        print("\n** on_train_epoch_end **")

    def on_validation_epoch_end(self):
        """Hook executed at the end of each validation epoch."""
        print("\n** on_validation_epoch_end **")

    def on_test_epoch_end(self):
        """Hook executed at the end of each test epoch."""
        print("\n** on_test_epoch_end **")

if __name__ == "__main__":
    # Define input shape (batch size, channels, depth, height, width)
    input_shape = (5, 3, 20, 256, 256)

    # Initialize the VideoPrivatizer model with a specific learning rate
    model = VideoPrivatizer(channels=input_shape[1], learning_rate=1e-4)

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
