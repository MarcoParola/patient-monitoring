import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from src.models.conv_backbone import CNN3DLightning
from src.models.skeleton.openpose.openpose_skeleton import OpenPoseAPI
from src.models.mlp import MLP
from src.models.LSTM import LSTM
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image as PILImage
import wandb
import os
from PIL import Image as PILImage

class PoseClassifier(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 640, 480), output_dim=9, backbone="CNN3D", end = "mlp", detect_face=False, dataset="Custom", detect_hands=False, fps=1):
        super(PoseClassifier, self).__init__()
        self.conv_backbone = None
        if (backbone == "YOLO"):
            self.feature_dim = 17 * 2 * fps * 5  # Number of keypoints for YOLO
        if (backbone == "CNN3D"):
            if dataset == "KTH":
                kth = True
            self.conv_backbone = CNN3DLightning(in_channels=input_shape[1], kth=kth)
            self.feature_dim = self.conv_backbone.feature_dim  # Number of features extracted by the backbone
            print(self.feature_dim)
        elif (backbone == "OpenPose"):
            self.conv_backbone = OpenPoseAPI(detect_face=detect_face, detect_hands=detect_hands, fps=fps)
            self.feature_dim = self.conv_backbone.feature_dim

        # MLP for final classification
        
        if (end == "mlp"):
            self.mlp = MLP(input_dim=self.feature_dim, output_dim=output_dim)
        elif (end == "lstm"):   
            self.lstm = LSTM(input_size=self.feature_dim, output_size=output_dim)

        self.test_outputs = [] 
        self.val_outputs = []
        self.output_dim = output_dim
        self.accumulated_val_preds = []
        self.accumulated_val_labels = []
        self.accumulated_test_preds = []
        self.accumulated_test_labels = []

    def forward(self, video_input):
        if (self.conv_backbone.__class__.__name__ == "CNN3DLightning"):
            video_input = self.conv_backbone(video_input)     
        elif (self.conv_backbone.__class__.__name__ != "OpenPoseAPI"):
            print("entro")
            video_input = video_input.flatten()
            print(video_input.shape)
         
        output = self.mlp(video_input)
        return output

    def _common_step(self, batch, step_type):
        video_input, labels = batch
        # Prediction
        pred = self(video_input)
        if pred.size(0) == 9:  # Dimensione pred per singolo esempio
            pred = pred.unsqueeze(0)  
        loss = F.cross_entropy(pred, labels)

        # Log metrics with WandB
        self.log(f"{step_type}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_accuracy", self.compute_accuracy(pred, labels), prog_bar=True, on_step=False, on_epoch=True)

        return {'loss': loss, 'logits': pred, 'labels': labels}
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        results = self._common_step(batch, step_type="val")
        self.accumulated_val_preds.append(torch.argmax(results["logits"], dim=1))
        self.accumulated_val_labels.append(results["labels"])
        return results

    def test_step(self, batch, batch_idx):
        results = self._common_step(batch, step_type="test")
        self.accumulated_test_preds.append(torch.argmax(results["logits"], dim=1))
        self.accumulated_test_labels.append(results["labels"])
        return results

    def on_validation_epoch_end(self):
        # Compute the confusion matrix after the entire validation epoch
        all_preds = torch.cat(self.accumulated_val_preds, dim=0)
        all_labels = torch.cat(self.accumulated_val_labels, dim=0)

        conf_matrix = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy(), labels=list(range(self.output_dim)))
        
        # Print confusion matrix to terminal
        print("Validation Confusion Matrix:")
        print(conf_matrix)

        # Plotting confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(self.output_dim)), yticklabels=list(range(self.output_dim)))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()

        # Save the plot as a PNG file
        plot_filename = "validation_confusion_matrix.png"
        plt.savefig(plot_filename)
        plt.close()

        # Open the saved plot using PIL, convert to RGB, and save it
        im = PILImage.open(plot_filename)
        rgb_im = im.convert("RGB")
        rgb_im.save("validation_confusion_matrix.jpg")

        # Log the image to wandb
        wandb.log({"Validation Confusion Matrix": wandb.Image("validation_confusion_matrix.jpg")})
        # cancella il file temporaneo
        os.remove("validation_confusion_matrix.jpg")

        # Clear the list of accumulated predictions and labels for the next epoch
        self.accumulated_val_preds.clear()
        self.accumulated_val_labels.clear()

    def on_test_epoch_end(self):
        # Compute the confusion matrix after the entire test epoch
        all_preds = torch.cat(self.accumulated_test_preds, dim=0)
        all_labels = torch.cat(self.accumulated_test_labels, dim=0)

        conf_matrix = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy(), labels=list(range(self.output_dim)))

        # Print confusion matrix to terminal
        print("Test Confusion Matrix:")
        print(conf_matrix)

        # Plotting confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(self.output_dim)), yticklabels=list(range(self.output_dim)))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Test Confusion Matrix")
        plt.tight_layout()

        # Save the plot as a PNG file
        plot_filename = "test_confusion_matrix.png"
        plt.savefig(plot_filename)
        plt.close()

        # Open the saved plot using PIL, convert to RGB, and save it
        im = PILImage.open(plot_filename)
        rgb_im = im.convert("RGB")
        rgb_im.save("test_confusion_matrix.jpg")

        # Log the image to wandb
        wandb.log({"Test Confusion Matrix": wandb.Image("test_confusion_matrix.jpg")})
        # cancella il file temporaneo
        os.remove("test_confusion_matrix.jpg")

        # Clear the list of accumulated predictions and labels for the next test

        self.accumulated_test_preds.clear()
        self.accumulated_test_labels.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def compute_accuracy(self, preds, labels):
        # Accuracy calculation
        _, predicted = torch.max(preds, 1)
        correct = (predicted == labels).float().sum()
        accuracy = correct / labels.size(0)
        return accuracy

