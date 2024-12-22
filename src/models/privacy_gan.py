import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from pose_classifier import PoseClassifier
from privacy_classifier import PrivacyClassifier
from video_privatizer import VideoPrivatizer

class PrivacyGAN(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 256, 256), output_dim_pose=9, output_dim_privacy=(4, 1, 1), alpha=1.0, beta=0.5):
        super(PrivacyGAN, self).__init__()

        # Generatore
        self.generator = VideoPrivatizer(channels=input_shape[1])

        # Discriminatori
        self.pose_classifier = PoseClassifier(input_shape, output_dim_pose)
        self.privacy_classifier = PrivacyClassifier(input_shape, output_dim_privacy)

        self.alpha = alpha
        self.beta = beta

    def forward(self, video_input):
        # Generatore produce video privatizzato
        privatized_video, _ = self.generator(video_input)
        return privatized_video

    def adversarial_loss(self, privatized_video, labels):
        # Loss del classificatore di pose
        pred_pose = self.pose_classifier(privatized_video)
        pose_loss = F.cross_entropy(pred_pose, labels['pose'])

        # Loss del classificatore di privacy
        pred_privacy = self.privacy_classifier(privatized_video)
        privacy_loss_skin = F.cross_entropy(pred_privacy[0], labels['skin_color'])
        privacy_loss_gender = F.binary_cross_entropy_with_logits(pred_privacy[1], labels['gender'].unsqueeze(1))
        privacy_loss_age = F.mse_loss(pred_privacy[2], labels['age'].float().unsqueeze(1))
        privacy_loss = privacy_loss_skin + privacy_loss_gender + privacy_loss_age

        # Loss combinata per il generatore
        generator_loss = self.alpha * pose_loss - self.beta * privacy_loss

        return pose_loss, privacy_loss, generator_loss

    def _common_step(self, batch, step_type):
        video_input, labels = batch

        # Generazione del video privatizzato
        privatized_video = self(video_input)

        # Calcolo delle loss
        pose_loss, privacy_loss, generator_loss = self.adversarial_loss(privatized_video, labels)

        # Logging
        self.log(f"{step_type}_pose_loss", pose_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_privacy_loss", privacy_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_generator_loss", generator_loss, prog_bar=True, on_step=True, on_epoch=True)

        return generator_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, step_type="test")

    def configure_optimizers(self):
        # Ottimizzatori separati per ogni modulo
        gen_optimizer = Adam(self.generator.parameters(), lr=1e-4)
        pose_optimizer = Adam(self.pose_classifier.parameters(), lr=1e-4)
        privacy_optimizer = Adam(self.privacy_classifier.parameters(), lr=1e-4)

        return [gen_optimizer, pose_optimizer, privacy_optimizer]

if __name__ == "__main__":
    # Configurazione di esempio
    input_shape = (5, 3, 20, 256, 256)  # Batch size di 5
    num_classes_pose = 9
    num_classes_privacy = (4, 1, 1)

    # Inizializzazione del modello
    model = PrivacyGAN(input_shape, num_classes_pose, num_classes_privacy)

    # Input fittizio
    video_input = torch.randn(input_shape)
    labels = {
        'pose': torch.randint(0, num_classes_pose, (input_shape[0],)),
        'skin_color': torch.randint(0, num_classes_privacy[0], (input_shape[0],)),
        'gender': torch.randint(0, 2, (input_shape[0],), dtype=torch.float32),
        'age': torch.randn(input_shape[0])
    }
    batch = (video_input, labels)

    # Test del metodo forward
    privatized_video = model(video_input)
    print(f"Privatized video shape: {privatized_video.shape}")

    # Test di training
    loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {loss}")
