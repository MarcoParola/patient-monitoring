import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.classification import ConfusionMatrix
from src.models.conv_backbone import CNN3DLightning
from src.models.mlp import MLP

# PrivacyClassifier combines a convolutional backbone with separate MLPs for each task
class PrivacyClassifier(pl.LightningModule):
    def __init__(self, channels, output_dim=(4, 1, 1), learning_rate=1e-4):
        """
        Initialize the PrivacyClassifier model.
        Args:
            channels (int): Number of input channels for the video input.
            output_dim (tuple): Number of output dimensions for skin color, gender, and age predictions.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(PrivacyClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.conv_backbone = CNN3DLightning(in_channels=channels)
        self.mlp_skin_color = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim[0], task_type='multiclass_classification')
        self.mlp_gender = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim[1], task_type='binary_classification')
        self.mlp_age = MLP(input_dim=self.conv_backbone.feature_output_dim, output_dim=output_dim[2], task_type='regression')

        # Confusion matrix metrics
        self.confusion_matrix_skin_color = ConfusionMatrix(task="multiclass", num_classes=output_dim[0])
        self.confusion_matrix_gender = ConfusionMatrix(task="binary")

        # Buffers for validation and test steps
        self.val_preds_skin_color = []
        self.val_labels_skin_color = []
        self.val_preds_gender = []
        self.val_labels_gender = []

    def forward(self, video_input):
        """
        Perform a forward pass through the model.
        Args:
            video_input (Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predictions for skin color, gender, and age.
        """
        x_video = self.conv_backbone(video_input)
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

        loss_skin_color = F.cross_entropy(pred_skin_color, labels['skin_color'])
        loss_gender = F.binary_cross_entropy_with_logits(pred_gender, labels['gender'].unsqueeze(1))
        loss_age = F.mse_loss(pred_age, labels['age'].unsqueeze(1).float())

        total_loss = loss_skin_color + loss_gender + loss_age

        if step_type in ["val", "test"]:
            self.val_preds_skin_color.append(torch.argmax(pred_skin_color, dim=1))
            self.val_labels_skin_color.append(labels['skin_color'])
            self.val_preds_gender.append((torch.sigmoid(pred_gender) > 0.5).squeeze(1))
            self.val_labels_gender.append(labels['gender'])

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        Returns:
            Optimizer: Adam optimizer with the specified learning rate.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_validation_epoch_end(self):
        self._log_confusion_matrices("Validation")

    def on_test_epoch_end(self):
        self._log_confusion_matrices("Test")

    def _log_confusion_matrices(self, step_type):
        # Skin Color Confusion Matrix
        all_preds_skin_color = torch.cat(self.val_preds_skin_color)
        all_labels_skin_color = torch.cat(self.val_labels_skin_color)
        conf_matrix_skin_color = self.confusion_matrix_skin_color(all_preds_skin_color, all_labels_skin_color)
        print(f"\n{step_type} Confusion Matrix for Skin Color:\n{conf_matrix_skin_color}")

        # Gender Confusion Matrix
        all_preds_gender = torch.cat(self.val_preds_gender)
        all_labels_gender = torch.cat(self.val_labels_gender)
        conf_matrix_gender = self.confusion_matrix_gender(all_preds_gender, all_labels_gender)
        print(f"\n{step_type} Confusion Matrix for Gender:\n{conf_matrix_gender}")

        # Clear buffers
        self.val_preds_skin_color.clear()
        self.val_labels_skin_color.clear()
        self.val_preds_gender.clear()
        self.val_labels_gender.clear()
    
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

    # Simulate the end of an epoch to print confusion matrices
    model.val_preds_skin_color.append(torch.argmax(output_skin_color, dim=1))
    model.val_labels_skin_color.append(labels['skin_color'])
    model.val_preds_gender.append((torch.sigmoid(output_gender) > 0.5).squeeze(1))
    model.val_labels_gender.append(labels['gender'])
    model.on_test_epoch_end()

    print("\nTest completed successfully!")
