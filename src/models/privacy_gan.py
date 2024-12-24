import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from src.models.video_privatizer import VideoPrivatizer
from src.models.pose_classifier import PoseClassifier
from src.models.privacy_classifier import PrivacyClassifier

class PrivacyGAN(pl.LightningModule):
    def __init__(self, input_shape, output_dim_pose, output_dim_privacy, alpha=1.0, beta=1.0):
        super(PrivacyGAN, self).__init__()
        
        # Set manual optimization
        self.automatic_optimization = False
        
        # Initialize components
        self.video_privatizer = VideoPrivatizer(channels=input_shape[1])
        self.pose_classifier = PoseClassifier(input_shape=input_shape, output_dim=output_dim_pose)
        self.privacy_classifier = PrivacyClassifier(input_shape=input_shape, output_dim=output_dim_privacy)
        
        # Loss weights
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        privatized_video, _ = self.video_privatizer(x)
        pose_pred = self.pose_classifier(privatized_video)
        privacy_preds = self.privacy_classifier(privatized_video)
        return privatized_video, pose_pred, privacy_preds

    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_privatizer, opt_pose, opt_privacy = self.optimizers()
        
        # Get input data
        x, labels = batch  # labels: [pose, skin_color, gender, age]
        
        # Step 1: Train Pose Classifier
        opt_pose.zero_grad()
        privatized_video, _ = self.video_privatizer(x)
        pose_pred = self.pose_classifier(privatized_video.detach())
        pose_loss = nn.functional.cross_entropy(pose_pred, labels[0])
        self.manual_backward(pose_loss)
        opt_pose.step()
        
        # Step 2: Train Privacy Classifier
        opt_privacy.zero_grad()
        privatized_video, _ = self.video_privatizer(x)
        privacy_preds = self.privacy_classifier(privatized_video.detach())
        
        privacy_loss_skin = nn.functional.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = nn.functional.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = nn.functional.mse_loss(
            privacy_preds[2], labels[3].unsqueeze(1).float()
        )
        privacy_loss = privacy_loss_skin + privacy_loss_gender + privacy_loss_age
        self.manual_backward(privacy_loss)
        opt_privacy.step()
        
        # Step 3: Train Privatizer
        opt_privatizer.zero_grad()
        privatized_video, _ = self.video_privatizer(x)
        pose_pred = self.pose_classifier(privatized_video)
        privacy_preds = self.privacy_classifier(privatized_video)
        
        # Calculate losses for privatizer update
        pose_loss_for_privatizer = nn.functional.cross_entropy(pose_pred, labels[0])
        
        privacy_loss_skin = nn.functional.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = nn.functional.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = nn.functional.mse_loss(
            privacy_preds[2], labels[3].unsqueeze(1).float()
        )
        privacy_loss_for_privatizer = privacy_loss_skin + privacy_loss_gender + privacy_loss_age
        
        # Combined loss for privatizer
        privatizer_loss = self.alpha * pose_loss_for_privatizer - self.beta * privacy_loss_for_privatizer
        self.manual_backward(privatizer_loss)
        opt_privatizer.step()
        
        # Log metrics
        self.log("train_pose_loss", pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_privacy_loss", privacy_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_privatizer_loss", privatizer_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Calculate and log accuracies
        with torch.no_grad():
            pose_acc = self.pose_classifier.compute_accuracy(pose_pred, labels[0])
            privacy_acc_skin = self.privacy_classifier.compute_accuracy_multi_class(
                privacy_preds[0], labels[1]
            )
            privacy_acc_gender = self.privacy_classifier.compute_accuracy_binary(
                privacy_preds[1], labels[2]
            )
            
            self.log("train_pose_acc", pose_acc, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_privacy_acc_skin", privacy_acc_skin, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_privacy_acc_gender", privacy_acc_gender, prog_bar=True, on_step=True, on_epoch=True)
        
        return privatizer_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        
        # Forward pass
        privatized_video, pose_pred, privacy_preds = self(x)
        
        # Calculate losses
        pose_loss = nn.functional.cross_entropy(pose_pred, labels[0])
        
        privacy_loss_skin = nn.functional.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = nn.functional.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = nn.functional.mse_loss(
            privacy_preds[2], labels[3].unsqueeze(1).float()
        )
        privacy_loss = privacy_loss_skin + privacy_loss_gender + privacy_loss_age
        
        # Combined loss
        total_loss = self.alpha * pose_loss - self.beta * privacy_loss
        
        # Log metrics
        self.log("val_pose_loss", pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_privacy_loss", privacy_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Calculate and log accuracies
        pose_acc = self.pose_classifier.compute_accuracy(pose_pred, labels[0])
        privacy_acc_skin = self.privacy_classifier.compute_accuracy_multi_class(
            privacy_preds[0], labels[1]
        )
        privacy_acc_gender = self.privacy_classifier.compute_accuracy_binary(
            privacy_preds[1], labels[2]
        )
        
        self.log("val_pose_acc", pose_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_privacy_acc_skin", privacy_acc_skin, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_privacy_acc_gender", privacy_acc_gender, prog_bar=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        
        # Forward pass
        privatized_video, pose_pred, privacy_preds = self(x)
        
        # Calculate losses
        pose_loss = nn.functional.cross_entropy(pose_pred, labels[0])
        
        privacy_loss_skin = nn.functional.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = nn.functional.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = nn.functional.mse_loss(
            privacy_preds[2], labels[3].unsqueeze(1).float()
        )
        privacy_loss = privacy_loss_skin + privacy_loss_gender + privacy_loss_age
        
        # Combined loss
        total_loss = self.alpha * pose_loss - self.beta * privacy_loss
        
        # Log metrics
        self.log("test_pose_loss", pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_privacy_loss", privacy_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Calculate and log accuracies
        pose_acc = self.pose_classifier.compute_accuracy(pose_pred, labels[0])
        privacy_acc_skin = self.privacy_classifier.compute_accuracy_multi_class(
            privacy_preds[0], labels[1]
        )
        privacy_acc_gender = self.privacy_classifier.compute_accuracy_binary(
            privacy_preds[1], labels[2]
        )
        
        self.log("test_pose_acc", pose_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_privacy_acc_skin", privacy_acc_skin, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_privacy_acc_gender", privacy_acc_gender, prog_bar=True, on_step=True, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
        opt_privatizer = Adam(self.video_privatizer.parameters(), lr=1e-4)
        opt_pose = Adam(self.pose_classifier.parameters(), lr=1e-4)
        opt_privacy = Adam(self.privacy_classifier.parameters(), lr=1e-4)
        return [opt_privatizer, opt_pose, opt_privacy]