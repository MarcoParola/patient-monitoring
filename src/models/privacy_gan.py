import torch
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.models.pose_classifier import PoseClassifier
from src.models.privacy_classifier import PrivacyClassifier

class PrivacyGAN(pl.LightningModule):
    def __init__(self, channels, output_dim_pose, output_dim_privacy, loss_weights, learning_rates, privacy_model_type="VIDEO_PRIVATIZER"):
        """
        Initialize PrivacyGAN.
        
        Args:
            channels (int): Number of input channels.
            output_dim_pose (int): Number of pose classification output dimensions.
            output_dim_privacy (tuple): Output dimensions for the privacy classifier.
            loss_weights (tuple): Loss weights for pose and privacy losses (alpha, beta).
            learning_rates (tuple): Learning rates for the privatizer, pose classifier, and privacy classifier.
        """
        super(PrivacyGAN, self).__init__()
        
        # Set manual optimization
        self.automatic_optimization = False
        
        # Initialize generator
        if privacy_model_type == "VIDEO_PRIVATIZER":
            from src.models.privacy_generator.video_privatizer import VideoPrivatizer
            self.video_privatizer = VideoPrivatizer(channels=channels)
            
        elif privacy_model_type == "STYLEGAN2":
            from src.models.privacy_generator.deep_privacy import StyleGAN2Privatizer
            self.video_privatizer = StyleGAN2Privatizer(channels=channels)
            
        elif privacy_model_type == "DEEP_PRIVACY2":
            from src.models.privacy_generator.deep_privacy import DeepPrivacy2Privatizer
            self.video_privatizer = DeepPrivacy2Privatizer(channels=channels)
            
        elif privacy_model_type == "BLUR":
            from src.models.privacy_generator.privacy_filter import BlurPrivacyFilter
            self.video_privatizer = BlurPrivacyFilter(channels=channels)
            
        elif privacy_model_type == "PIXELATE":
            from src.models.privacy_generator.privacy_filter import PixelatePrivacyFilter
            self.video_privatizer = PixelatePrivacyFilter()
        
        # Initilize discriminators
        self.pose_classifier = PoseClassifier(channels=channels, output_dim=output_dim_pose)
        self.privacy_classifier = PrivacyClassifier(channels=channels, output_dim=output_dim_privacy)
        
        # Loss weights
        self.alpha = loss_weights[0]
        self.beta = loss_weights[1]

        # Learning rates for optimizers
        self.lr_privatizer = learning_rates[0]
        self.lr_pose = learning_rates[1]
        self.lr_privacy = learning_rates[2]

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
        pose_loss = F.cross_entropy(pose_pred, labels[0])
        self.manual_backward(pose_loss)
        opt_pose.step()
        
        # Step 2: Train Privacy Classifier
        opt_privacy.zero_grad()
        privatized_video, _ = self.video_privatizer(x)
        privacy_preds = self.privacy_classifier(privatized_video.detach())
        
        privacy_loss_skin = F.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = F.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = F.mse_loss(
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
        pose_loss_for_privatizer = F.cross_entropy(pose_pred, labels[0])
        
        privacy_loss_skin = F.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = F.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = F.mse_loss(
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
        _, pose_pred, privacy_preds = self(x)
        
        # Calculate losses
        pose_loss = F.cross_entropy(pose_pred, labels[0])
        
        privacy_loss_skin = F.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = F.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = F.mse_loss(
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
        _, pose_pred, privacy_preds = self(x)
        
        # Calculate losses
        pose_loss = F.cross_entropy(pose_pred, labels[0])
        
        privacy_loss_skin = F.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = F.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = F.mse_loss(
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
        """
        Configure optimizers for the model components.
        
        Returns:
            list: List of Adam optimizers for each component.
        """
        opt_privatizer = Adam(self.video_privatizer.parameters(), lr=self.lr_privatizer)
        opt_pose = Adam(self.pose_classifier.parameters(), lr=self.lr_pose)
        opt_privacy = Adam(self.privacy_classifier.parameters(), lr=self.lr_privacy)
        return [opt_privatizer, opt_pose, opt_privacy]
    
    def configure_callbacks(self):
        """
        Configure early stopping and model checkpoint callbacks.
        """
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_total_loss",
            patience=10,
            mode="min",
            verbose=True
        )

        # Model checkpoint callback
        model_checkpoint = ModelCheckpoint(
            monitor="val_total_loss",
            dirpath="checkpoints/",
            filename="best-model-{epoch:02d}-{val_total_loss:.4f}",
            save_top_k=1,
            mode="min",
            verbose=True
        )

        return [early_stopping, model_checkpoint]
