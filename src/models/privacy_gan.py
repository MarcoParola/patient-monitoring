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

        # Initialize lists to store predictions and labels for confusion matrices
        self.val_pose_preds = []
        self.val_pose_labels = []
        self.val_privacy_skin_preds = []
        self.val_privacy_skin_labels = []
        self.val_privacy_gender_preds = []
        self.val_privacy_gender_labels = []
        
        self.test_pose_preds = []
        self.test_pose_labels = []
        self.test_privacy_skin_preds = []
        self.test_privacy_skin_labels = []
        self.test_privacy_gender_preds = []
        self.test_privacy_gender_labels = []

    def forward(self, x):
        privatized_video, _ = self.video_privatizer(x)
        pose_pred = self.pose_classifier(privatized_video)
        privacy_preds = self.privacy_classifier(privatized_video)
        return privatized_video, pose_pred, privacy_preds

    def _store_predictions(self, pose_pred, privacy_preds, labels, phase="val"):
        # Get predicted classes
        pose_pred_class = torch.argmax(pose_pred, dim=1)
        privacy_skin_pred_class = torch.argmax(privacy_preds[0], dim=1)
        privacy_gender_pred = (torch.sigmoid(privacy_preds[1]) > 0.5).squeeze(1)
        
        # Store predictions and labels based on phase
        if phase == "val":
            self.val_pose_preds.append(pose_pred_class)
            self.val_pose_labels.append(labels[0])
            self.val_privacy_skin_preds.append(privacy_skin_pred_class)
            self.val_privacy_skin_labels.append(labels[1])
            self.val_privacy_gender_preds.append(privacy_gender_pred)
            self.val_privacy_gender_labels.append(labels[2])
        else:  # test
            self.test_pose_preds.append(pose_pred_class)
            self.test_pose_labels.append(labels[0])
            self.test_privacy_skin_preds.append(privacy_skin_pred_class)
            self.test_privacy_skin_labels.append(labels[1])
            self.test_privacy_gender_preds.append(privacy_gender_pred)
            self.test_privacy_gender_labels.append(labels[2])

    def _compute_confusion_matrices(self, phase="val"):
        if phase == "val":
            preds_lists = [self.val_pose_preds, self.val_privacy_skin_preds, self.val_privacy_gender_preds]
            labels_lists = [self.val_pose_labels, self.val_privacy_skin_labels, self.val_privacy_gender_labels]
            prefix = "val"
        else:
            preds_lists = [self.test_pose_preds, self.test_privacy_skin_preds, self.test_privacy_gender_preds]
            labels_lists = [self.test_pose_labels, self.test_privacy_skin_labels, self.test_privacy_gender_labels]
            prefix = "test"
        
        # Compute confusion matrices
        pose_matrix = confusion_matrix(
            torch.cat(preds_lists[0]),
            torch.cat(labels_lists[0]),
            task="multiclass",
            num_classes=self.pose_classifier.output_dim
        )
        
        privacy_skin_matrix = self.privacy_classifier.confusion_matrix_skin_color(
            torch.cat(preds_lists[1]),
            torch.cat(labels_lists[1])
        )
        
        privacy_gender_matrix = self.privacy_classifier.confusion_matrix_gender(
            torch.cat(preds_lists[2]),
            torch.cat(labels_lists[2])
        )
        
        # Log metrics derived from confusion matrices
        pose_accuracy = torch.diagonal(pose_matrix).sum() / pose_matrix.sum()
        privacy_skin_accuracy = torch.diagonal(privacy_skin_matrix).sum() / privacy_skin_matrix.sum()
        privacy_gender_accuracy = torch.diagonal(privacy_gender_matrix).sum() / privacy_gender_matrix.sum()
        
        self.log(f"{prefix}_pose_conf_accuracy", pose_accuracy)
        self.log(f"{prefix}_privacy_skin_conf_accuracy", privacy_skin_accuracy)
        self.log(f"{prefix}_privacy_gender_conf_accuracy", privacy_gender_accuracy)
        
        # For each class in pose matrix
        n_pose_classes = pose_matrix.size(0)
        for i in range(n_pose_classes):
            precision = pose_matrix[i, i] / (pose_matrix[:, i].sum() + 1e-8)
            recall = pose_matrix[i, i] / (pose_matrix[i, :].sum() + 1e-8)
            self.log(f"{prefix}_pose_class{i}_precision", precision)
            self.log(f"{prefix}_pose_class{i}_recall", recall)
        
        # Log confusion matrices as figures using torchmetrics plotting functionality
        if self.logger is not None:
            # Convert matrices to numpy for visualization
            pose_matrix_np = pose_matrix.cpu().numpy()
            privacy_skin_matrix_np = privacy_skin_matrix.cpu().numpy()
            privacy_gender_matrix_np = privacy_gender_matrix.cpu().numpy()
            
            # Print matrices for visualization in console
            print(f"\n{phase.capitalize()} Pose Confusion Matrix:")
            print(pose_matrix_np)
            print(f"\n{phase.capitalize()} Privacy Skin Color Confusion Matrix:")
            print(privacy_skin_matrix_np)
            print(f"\n{phase.capitalize()} Privacy Gender Confusion Matrix:")
            print(privacy_gender_matrix_np)

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

        # Calculate and log privacy accuracies during training
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

        # Store predictions and labels for confusion matrices
        self._store_predictions(pose_pred, privacy_preds, labels, phase=phase)

        # Calculate and log accuracies
        with torch.no_grad():
            pose_acc = self.pose_classifier.compute_accuracy(pose_pred, labels[0])
            privacy_acc_skin = self.privacy_classifier.compute_accuracy_multi_class(
                privacy_preds[0], labels[1]
            )
            privacy_acc_gender = self.privacy_classifier.compute_accuracy_binary(
                privacy_preds[1], labels[2]
            )

        # Logga le metriche
        self.log(f"{phase}_pose_loss", pose_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_privacy_loss", privacy_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_pose_acc", pose_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_privacy_acc_skin", privacy_acc_skin, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_privacy_acc_gender", privacy_acc_gender, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, phase="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, phase="test")

    def on_validation_epoch_end(self):
        """Compute and log confusion matrices at the end of validation epoch"""
        self._compute_confusion_matrices(phase="val")
        # Clear stored predictions and labels
        self.val_pose_preds.clear()
        self.val_pose_labels.clear()
        self.val_privacy_skin_preds.clear()
        self.val_privacy_skin_labels.clear()
        self.val_privacy_gender_preds.clear()
        self.val_privacy_gender_labels.clear()

    def on_test_epoch_end(self):
        """Compute and log confusion matrices at the end of test epoch"""
        self._compute_confusion_matrices(phase="test")
        # Clear stored predictions and labels
        self.test_pose_preds.clear()
        self.test_pose_labels.clear()
        self.test_privacy_skin_preds.clear()
        self.test_privacy_skin_labels.clear()
        self.test_privacy_gender_preds.clear()
        self.test_privacy_gender_labels.clear()

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
    
    def configure_callbacks(self, enable_early_stopping=True, patience=10):
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
                monitor="val_total_loss",  # 
                patience=patience,         # 
                mode="min",                # 
                verbose=True
            )
            callbacks.append(early_stopping)

        return callbacks

