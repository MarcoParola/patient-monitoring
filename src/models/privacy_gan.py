import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from src.models.pose_classifier import PoseClassifier
from src.models.privacy_classifier import PrivacyClassifier
from src.models.video_privatizer import VideoPrivatizer

class Conv3DModel(nn.Module):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Define a 3D convolutional layer with a 1x1x1 kernel
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv3d(x)
        return x, x

class PrivacyGAN(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 256, 256), output_dim_pose=9, 
                 output_dim_privacy=(4, 1, 1), alpha=1.0, beta=0.5):
        super(PrivacyGAN, self).__init__()
        
        #self.generator = PrivacyClassifier(channels=input_shape[1])
        self.generator = Conv3DModel()
        self.pose_classifier = PoseClassifier(input_shape, output_dim_pose)
        self.privacy_classifier = PrivacyClassifier(input_shape, output_dim_privacy)
        
        self.alpha = alpha
        self.beta = beta
        self.automatic_optimization = False

    def forward(self, video_input):
        if not video_input.requires_grad:
            video_input = video_input.detach().requires_grad_(True)
        privatized_video, _ = self.generator(video_input)
        return privatized_video

    def adversarial_loss(self, privatized_video, labels):
        # Compute pose loss with original video
        pred_pose = self.pose_classifier(privatized_video)
        pose_loss = F.cross_entropy(pred_pose, labels[0])

        # Compute privacy losses
        pred_privacy = self.privacy_classifier(privatized_video)
        
        privacy_loss_skin = F.cross_entropy(
            pred_privacy[0], 
            labels[1]
        )
        
        privacy_loss_gender = F.binary_cross_entropy_with_logits(
            pred_privacy[1],
            labels[2].unsqueeze(1)
        )

        privacy_loss_age = F.mse_loss(
            pred_privacy[2],
            labels[3].unsqueeze(1).float()
        )
        
        privacy_loss = privacy_loss_skin + privacy_loss_gender + privacy_loss_age
        generator_loss = self.alpha * pose_loss - self.beta * privacy_loss

        return pose_loss, privacy_loss, generator_loss

    def common_step(self, batch, batch_idx, stage='train'):
        video_input, labels = batch
        
        # Generate privatized video with gradient tracking
        privatized_video = self(video_input)
        
        # Calculate losses
        pose_loss, privacy_loss, generator_loss = self.adversarial_loss(
            privatized_video, 
            labels
        )
        
        # Log metrics
        self.log(f"{stage}_pose_loss", pose_loss.item(), prog_bar=True, 
                on_step=(stage=='train'), on_epoch=True)
        self.log(f"{stage}_privacy_loss", privacy_loss.item(), prog_bar=True, 
                on_step=(stage=='train'), on_epoch=True)
        self.log(f"{stage}_generator_loss", generator_loss.item(), prog_bar=True, 
                on_step=(stage=='train'), on_epoch=True)
        
        return pose_loss, privacy_loss, generator_loss, privatized_video

    def training_step(self, batch, batch_idx):

        torch.autograd.set_detect_anomaly(True)

        # Get losses and generated video
        pose_loss, privacy_loss, generator_loss, _ = self.common_step(
            batch, 
            batch_idx, 
            'train'
        )
        
        # Get optimizers
        opt_gen, opt_pose, opt_privacy = self.optimizers()
        
        # Update generator
        opt_gen.zero_grad(set_to_none=True)
        self.manual_backward(generator_loss, retain_graph=True)
        opt_gen.step()
        
        # Update pose classifier
        opt_pose.zero_grad(set_to_none=True)
        self.manual_backward(pose_loss, retain_graph=True)
        opt_pose.step()
        
        # Update privacy classifier - last backward pass doesn't need retain_graph
        opt_privacy.zero_grad(set_to_none=True)
        self.manual_backward(privacy_loss)
        opt_privacy.step()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            self.common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        gen_optimizer = Adam(self.generator.parameters(), lr=1e-4)
        pose_optimizer = Adam(self.pose_classifier.parameters(), lr=1e-4)
        privacy_optimizer = Adam(self.privacy_classifier.parameters(), lr=1e-4)
        return [gen_optimizer, pose_optimizer, privacy_optimizer]

if __name__ == "__main__":
    input_shape = (5, 3, 20, 256, 256)
    num_classes_pose = 9
    num_classes_privacy = (4, 1, 1)
    
    model = PrivacyGAN(input_shape, num_classes_pose, num_classes_privacy)
    
    video_input = torch.randn(input_shape)
    labels = {
        'pose': torch.randint(0, num_classes_pose, (input_shape[0],)),
        'skin_color': torch.randint(0, num_classes_privacy[0], (input_shape[0],)),
        'gender': torch.randint(0, 2, (input_shape[0],), dtype=torch.float32),
        'age': torch.randn(input_shape[0])
    }
    batch = (video_input, labels)
    
    privatized_video = model(video_input)
    print(f"Privatized video shape: {privatized_video.shape}")
    model.training_step(batch, batch_idx=0)
    print("Training step completed.")