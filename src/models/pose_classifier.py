import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from src.models.conv_backbone import CNN3DLightning
from src.models.mlp import MLP

# PoseClassifier combines a convolutional backbone with an MLP for pose classification
class PoseClassifier(pl.LightningModule):
    def __init__(self, channels, output_dim=9, learning_rate=1e-4):
        """
        Initialize the PoseClassifier model.
        Args:
            channels (int): Number of input channels for the video input.
            output_dim (int): Number of output classes for classification.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(PoseClassifier, self).__init__()

        # Convolutional backbone for feature extraction
        self.conv_backbone = CNN3DLightning(in_channels=channels)

        # Multi-Layer Perceptron (MLP) for final classification
        self.mlp = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim)

        # Save the learning rate
        self.learning_rate = learning_rate

    def forward(self, video_input):
        """
        Perform a forward pass through the model.
        Args:
            video_input (Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        Returns:
            Tensor: Output logits of shape (batch_size, output_dim).
        """
        # Extract features using the convolutional backbone
        x_video = self.conv_backbone(video_input)
        # Apply the MLP to the extracted features for classification
        output = self.mlp(x_video)
        return output

    def _common_step(self, batch, step_type):
        """
        Shared logic for training, validation, and test steps.
        Args:
            batch (tuple): Tuple containing video inputs and labels.
            step_type (str): Type of step ('train', 'val', or 'test').
        Returns:
            dict: Loss and logits from the step.
        """
        video_input, labels = batch

        # Perform a forward pass and compute predictions
        pred = self(video_input)

        # Compute cross-entropy loss
        loss = F.cross_entropy(pred, labels)

        # Log loss and accuracy metrics
        self.log(f"{step_type}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_accuracy", self.compute_accuracy(pred, labels), prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'logits': pred}

    def training_step(self, batch, batch_idx):
        """
        Training step logic.
        Args:
            batch (tuple): Training batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Loss and logits for the current batch.
        """
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step logic.
        Args:
            batch (tuple): Validation batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Loss and logits for the current batch.
        """
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        """
        Test step logic.
        Args:
            batch (tuple): Test batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Loss and logits for the current batch.
        """
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        Returns:
            Optimizer: Adam optimizer with the specified learning rate.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        """Hook executed at the end of each training epoch."""
        print("\n** on_train_epoch_end **")

    def on_validation_epoch_end(self):
        """Hook executed at the end of each validation epoch."""
        print("\n** on_validation_epoch_end **")

    def on_test_epoch_end(self):
        """Hook executed at the end of each test epoch."""
        print("\n** on_test_epoch_end **")

    def compute_accuracy(self, preds, labels):
        """
        Compute the accuracy of predictions.
        Args:
            preds (Tensor): Logits predicted by the model.
            labels (Tensor): Ground truth labels.
        Returns:
            Tensor: Accuracy score as a scalar tensor.
        """
        _, predicted = torch.max(preds, 1)  # Get the predicted class
        correct = (predicted == labels).float().sum()  # Count correct predictions
        accuracy = correct / labels.size(0)  # Compute accuracy
        return accuracy

if __name__ == "__main__":
    print("Test: PoseClassifier")

    # Configuration
    input_shape = (5, 3, 20, 256, 256)  # Batch size of 5
    num_classes = 9  # Number of output classes
    learning_rate = 0.001  # Specify a custom learning rate
    model = PoseClassifier(channels=input_shape[1], output_dim=num_classes, learning_rate=learning_rate)

    # Generate random input data for testing
    video_input = torch.randn(input_shape)  # Random video input
    labels = torch.randint(0, num_classes, (input_shape[0],))  # Random labels
    batch = (video_input, labels)

    # Test the forward method
    output = model(video_input)
    print(f"Output shape: {output.shape}")

    # Test the training step
    train_loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {train_loss['loss'].item()}")

    # Test the validation step
    val_loss = model.validation_step(batch, batch_idx=0)
    print(f"Validation loss: {val_loss['loss'].item()}")

    # Test the test step
    test_loss = model.test_step(batch, batch_idx=0)
    print(f"Test loss: {test_loss['loss'].item()}")

    print("\nTest completed successfully!")
