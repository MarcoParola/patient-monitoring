import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from src.models.conv_backbone import CNN3DLightning
from src.models.mlp import MLP

# PrivacyClassifier combines a convolutional backbone with separate MLPs for each task
class PrivacyClassifier(pl.LightningModule):
    def __init__(self, channels, output_dim=(4,1,1), learning_rate=1e-4):
        """
        Initialize the PrivacyClassifier model.
        Args:
            channels (int): Number of input channels for the video input.
            output_dim (tuple): Number of output dimensions for skin color, gender, and age predictions.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(PrivacyClassifier, self).__init__()
        
        # Store the learning rate
        self.learning_rate = learning_rate
        
        # Convolutional backbone for feature extraction
        self.conv_backbone = CNN3DLightning(in_channels=channels)
        
        # Separate MLPs for each metadata task
        self.mlp_skin_color = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim[0])
        self.mlp_gender = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim[1])
        self.mlp_age = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim[2])

    def forward(self, video_input):
        """
        Perform a forward pass through the model.
        Args:
            video_input (Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predictions for skin color, gender, and age.
        """
        # Extract features using the convolutional backbone
        x_video = self.conv_backbone(video_input)
        
        # Predict each metadata task independently
        output_skin_color = self.mlp_skin_color(x_video)
        output_gender = self.mlp_gender(x_video)
        output_age = self.mlp_age(x_video)

        return output_skin_color, output_gender, output_age

    def _common_step(self, batch, step_type):
        """
        Shared logic for training, validation, and test steps.
        Args:
            batch (tuple): Tuple containing video inputs and labels.
            step_type (str): Type of step ('train', 'val', or 'test').
        Returns:
            Tensor: Total loss for the step.
        """
        video_input, labels = batch
        pred_skin_color, pred_gender, pred_age = self(video_input)

        # Compute loss for each task
        loss_skin_color = F.cross_entropy(pred_skin_color, labels['skin_color'])
        loss_gender = F.binary_cross_entropy_with_logits(pred_gender, labels['gender'].unsqueeze(1))
        loss_age = F.mse_loss(pred_age, labels['age'].unsqueeze(1).float())

        # Sum the losses to compute total loss
        total_loss = loss_skin_color + loss_gender + loss_age

        # Compute accuracy for skin color and gender predictions
        acc_skin_color = self.compute_accuracy_multi_class(pred_skin_color, labels['skin_color'])
        acc_gender = self.compute_accuracy_binary(pred_gender, labels['gender'])

        # Log losses and accuracies
        self.log(f"{step_type}_loss_skin_color", loss_skin_color, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_loss_gender", loss_gender, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_loss_age", loss_age, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)

        self.log(f"{step_type}_acc_skin_color", acc_skin_color, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_acc_gender", acc_gender, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        """
        Training step logic.
        Args:
            batch (tuple): Training batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: Total loss for the training step.
        """
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step logic.
        Args:
            batch (tuple): Validation batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: Total loss for the validation step.
        """
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        """
        Test step logic.
        Args:
            batch (tuple): Test batch data (inputs and labels).
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: Total loss for the test step.
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

    def compute_accuracy_multi_class(self, preds, labels):
        """
        Compute accuracy for multi-class predictions.
        Args:
            preds (Tensor): Logits predicted by the model for skin color.
            labels (Tensor): Ground truth labels for skin color.
        Returns:
            Tensor: Accuracy score as a scalar tensor.
        """
        _, predicted = torch.max(preds, 1)  # Get predicted class
        correct = (predicted == labels).float().sum()  # Count correct predictions
        return correct / labels.size(0)

    def compute_accuracy_binary(self, preds, labels):
        """
        Compute accuracy for binary predictions.
        Args:
            preds (Tensor): Logits predicted by the model for gender.
            labels (Tensor): Ground truth binary labels for gender.
        Returns:
            Tensor: Accuracy score as a scalar tensor.
        """
        predicted = (torch.sigmoid(preds) > 0.5).float()  # Convert logits to binary predictions
        correct = (predicted == labels).float().sum()  # Count correct predictions
        return correct / labels.size(0)

if __name__ == "__main__":
    print("Test: PrivacyClassifier")
    
    # Configuration
    input_shape = (5, 3, 20, 256, 256)  # Batch size of 5
    num_classes = (4, 1, 1)  # Number of classes for skin color, gender, and age
    model = PrivacyClassifier(channels=input_shape[1], output_dim=num_classes)
    
    # Generate random input data for testing
    video_input = torch.randn(input_shape)  # Random video input
    labels = {
        'skin_color': torch.randint(0, num_classes[0], (input_shape[0],)),        # Classes for skin color
        'gender': torch.randint(0, 2, (input_shape[0],), dtype=torch.float32),    # Binary values for gender
        'age': torch.randn(input_shape[0])                                       # Continuous values for age
    }
    batch = (video_input, labels)
    
    # Test the forward method
    output_skin_color, output_gender, output_age = model(video_input)
    print(f"Output Skin Color shape: {output_skin_color.shape}")
    print(f"Output Gender shape: {output_gender.shape}")
    print(f"Output Age shape: {output_age.shape}")

    # Test the training_step
    train_loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {train_loss.item()}")

    # Test the validation_step
    val_loss = model.validation_step(batch, batch_idx=0)
    print(f"Validation loss: {val_loss.item()}")

    # Test the test_step
    test_loss = model.test_step(batch, batch_idx=0)
    print(f"Test loss: {test_loss.item()}")

    print("\nTest completed successfully!")
