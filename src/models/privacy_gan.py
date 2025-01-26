import torch
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn.functional as F
from torchmetrics.functional import confusion_matrix
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
        
        # Loss weights for pose and privacy losses
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
                      
        return privatizer_loss
    
    def _common_step(self, batch, phase):
        x, labels = batch
        privatized_video, pose_pred, privacy_preds = self(x)

        # Compute losses
        pose_loss = F.cross_entropy(pose_pred, labels[0])
        privacy_loss_skin = F.cross_entropy(privacy_preds[0], labels[1])
        privacy_loss_gender = F.binary_cross_entropy_with_logits(
            privacy_preds[1], labels[2].unsqueeze(1)
        )
        privacy_loss_age = F.mse_loss(
            privacy_preds[2], labels[3].unsqueeze(1).float()
        )
        privacy_loss = privacy_loss_skin + privacy_loss_gender + privacy_loss_age
        total_loss = self.alpha * pose_loss - self.beta * privacy_loss

        # Logga le metriche
        self.log(f"{phase}_pose_loss", pose_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_privacy_loss", privacy_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, phase="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, phase="test")

    def on_validation_epoch_end(self):
        """Compute and log accuracies and confusion matrices at the end of validation epoch"""
        all_pose_preds, all_pose_labels = [], []
        all_skin_preds, all_skin_labels = [], []
        all_gender_preds, all_gender_labels = [], []
        all_age_preds, all_age_labels = [], []

        with torch.no_grad():
            for batch in self.trainer.val_dataloaders:
                x, labels = batch
                _, pose_pred, privacy_preds = self(x)
                
                # Pose predictions
                pose_preds = torch.argmax(pose_pred, dim=1)
                all_pose_preds.append(pose_preds)
                all_pose_labels.append(labels[0])
                
                # Skin color predictions
                skin_preds = torch.argmax(privacy_preds[0], dim=1)
                all_skin_preds.append(skin_preds)
                all_skin_labels.append(labels[1])
                
                # Gender predictions
                gender_preds = (torch.sigmoid(privacy_preds[1]) > 0.5).float()
                all_gender_preds.append(gender_preds.squeeze())
                all_gender_labels.append(labels[2])
                
                # Age predictions (regression)
                age_preds = privacy_preds[2].squeeze()
                all_age_preds.append(age_preds)
                all_age_labels.append(labels[3])

        # Concatenate all predictions and labels
        all_pose_preds = torch.cat(all_pose_preds)
        all_pose_labels = torch.cat(all_pose_labels)
        all_skin_preds = torch.cat(all_skin_preds)
        all_skin_labels = torch.cat(all_skin_labels)
        all_gender_preds = torch.cat(all_gender_preds)
        all_gender_labels = torch.cat(all_gender_labels)
        all_age_preds = torch.cat(all_age_preds)
        all_age_labels = torch.cat(all_age_labels)

        # Compute accuracies
        pose_accuracy = (all_pose_preds == all_pose_labels).float().mean()
        skin_accuracy = (all_skin_preds == all_skin_labels).float().mean()
        gender_accuracy = (all_gender_preds == all_gender_labels).float().mean()
        age_mse = F.mse_loss(all_age_preds, all_age_labels.float())

        # Compute confusion matrices
        pose_cm = confusion_matrix(all_pose_preds, all_pose_labels, task='multiclass', num_classes=self.pose_classifier.output_dim)
        skin_cm = confusion_matrix(all_skin_preds, all_skin_labels, task='multiclass', num_classes=privacy_preds[0].shape[1])
        gender_cm = confusion_matrix(all_gender_preds, all_gender_labels, task='binary')

        print("Val Pose confusion matrix:")
        print(pose_cm)
        print("Val Skin color confusion matrix:")
        print(skin_cm)
        print("Val Gender confusion matrix:")
        print(gender_cm)
        print("Val Age MSE:", age_mse)

        # Log metrics
        self.log("val_pose_accuracy", pose_accuracy, prog_bar=True, on_epoch=True)
        self.log("val_skin_accuracy", skin_accuracy, prog_bar=True, on_epoch=True)
        self.log("val_gender_accuracy", gender_accuracy, prog_bar=True, on_epoch=True)
        self.log("val_age_mse", age_mse, prog_bar=True, on_epoch=True)

    def on_test_epoch_end(self):
        """Compute and log accuracies and confusion matrices at the end of test epoch"""
        all_pose_preds, all_pose_labels = [], []
        all_skin_preds, all_skin_labels = [], []
        all_gender_preds, all_gender_labels = [], []
        all_age_preds, all_age_labels = [], []

        with torch.no_grad():
            for batch in self.trainer.test_dataloaders:
                x, labels = batch
                _, pose_pred, privacy_preds = self(x)
                
                # Pose predictions
                pose_preds = torch.argmax(pose_pred, dim=1)
                all_pose_preds.append(pose_preds)
                all_pose_labels.append(labels[0])
                
                # Skin color predictions
                skin_preds = torch.argmax(privacy_preds[0], dim=1)
                all_skin_preds.append(skin_preds)
                all_skin_labels.append(labels[1])
                
                # Gender predictions
                gender_preds = (torch.sigmoid(privacy_preds[1]) > 0.5).float()
                all_gender_preds.append(gender_preds.squeeze())
                all_gender_labels.append(labels[2])
                
                # Age predictions (regression)
                age_preds = privacy_preds[2].squeeze()
                all_age_preds.append(age_preds)
                all_age_labels.append(labels[3])

        # Concatenate all predictions and labels
        all_pose_preds = torch.cat(all_pose_preds)
        all_pose_labels = torch.cat(all_pose_labels)
        all_skin_preds = torch.cat(all_skin_preds)
        all_skin_labels = torch.cat(all_skin_labels)
        all_gender_preds = torch.cat(all_gender_preds)
        all_gender_labels = torch.cat(all_gender_labels)
        all_age_preds = torch.cat(all_age_preds)
        all_age_labels = torch.cat(all_age_labels)

        # Compute accuracies
        pose_accuracy = (all_pose_preds == all_pose_labels).float().mean()
        skin_accuracy = (all_skin_preds == all_skin_labels).float().mean()
        gender_accuracy = (all_gender_preds == all_gender_labels).float().mean()
        age_mse = F.mse_loss(all_age_preds, all_age_labels.float())

        # Compute confusion matrices
        pose_cm = confusion_matrix(all_pose_preds, all_pose_labels, task='multiclass', num_classes=self.pose_classifier.output_dim)
        skin_cm = confusion_matrix(all_skin_preds, all_skin_labels, task='multiclass', num_classes=privacy_preds[0].shape[1])
        gender_cm = confusion_matrix(all_gender_preds, all_gender_labels, task='binary')

        print("Test Pose confusion matrix:")
        print(pose_cm)
        print("Test Skin color confusion matrix:")
        print(skin_cm)
        print("Test Gender confusion matrix:")
        print(gender_cm)
        print("Test Age MSE:", age_mse)

        # Log metrics
        self.log("test_pose_accuracy", pose_accuracy, prog_bar=True, on_epoch=True)
        self.log("test_skin_accuracy", skin_accuracy, prog_bar=True, on_epoch=True)
        self.log("test_gender_accuracy", gender_accuracy, prog_bar=True, on_epoch=True)
        self.log("test_age_mse", age_mse, prog_bar=True, on_epoch=True)        

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
    
    def configure_callbacks(self, enable_early_stopping=False, patience=30):
        """
        Configure callbacks for early stopping and model saving.

        Args:
            enable_early_stopping (bool): Flag to enable or disable early stopping.
            patience (int): Number of epochs without improvement before stopping training.

        Returns:
            list: List of configured callbacks.
        """
        callbacks = []

        if enable_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_total_loss",   # Monitor total loss on validation set
                patience=patience,          # Stop training after 10 epochs without improvement
                mode="min",                 # Minimize the monitored quantity
                verbose=True
            )
            callbacks.append(early_stopping)

        return callbacks

