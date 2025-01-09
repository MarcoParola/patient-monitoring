import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

class StyleGAN2Privatizer(pl.LightningModule):
    def __init__(self, channels, style_dim=512, n_mlp=8, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.channels = channels
        
        # Mapping network
        layers = [nn.Linear(style_dim, style_dim)]
        for _ in range(n_mlp - 1):
            layers.append(nn.Linear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        
        # Modified architecture to properly handle 3D data
        # Process each frame independently first
        self.frame_processor = nn.Sequential(
            nn.Conv2d(channels, channels * 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 16, channels * 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 8, channels * 4, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 4, channels, 3, padding=1),
        )
        
        # StyleGAN2 main blocks
        self.input = ConstantInput(channels * 16)
        self.conv1 = StyledConv(channels * 16, channels * 16, 3, style_dim)
        self.conv2 = StyledConv(channels * 16, channels * 8, 3, style_dim)
        self.conv3 = StyledConv(channels * 8, channels * 4, 3, style_dim)
        self.conv4 = StyledConv(channels * 4, channels, 3, style_dim)
        
        # Noise injection
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.noise3 = NoiseInjection()
        self.noise4 = NoiseInjection()
        
        # Temporal processing
        self.temporal_conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x):
        batch_size, c, d, h, w = x.size()
        
        # Generate random style vector
        style = torch.randn(batch_size, 512)
        style = self.mapping(style)
        
        # Process each frame independently first
        frame_features = []
        for i in range(d):
            frame = x[:, :, i, :, :]  # Get single frame
            frame_processed = self.frame_processor(frame)
            frame_features.append(frame_processed)
        
        # Stack processed frames back together
        x_processed = torch.stack(frame_features, dim=2)  # B, C, D, H, W
        
        # Apply temporal convolution
        out = self.temporal_conv(x_processed)
        
        # Process a representative middle frame through StyleGAN2 blocks
        mid_frame = out[:, :, d//2, :, :]
        styled = self.input(mid_frame)
        styled = self.noise1(self.conv1(styled, style))
        styled = self.noise2(self.conv2(styled, style))
        styled = self.noise3(self.conv3(styled, style))
        styled = self.noise4(self.conv4(styled, style))
        
        # Combine temporal and styled features
        out = out + styled.unsqueeze(2).expand(-1, -1, d, -1, -1)
        
        return out, self.temporal_conv(x)  # Return privatized output and identity features

    def _common_step(self, batch, batch_idx, step_type):
        x, _ = batch
        privatized, features = self(x)
        
        # Calculate losses
        recon_loss = F.mse_loss(privatized, x)
        privacy_loss = torch.mean(torch.abs(privatized - x))
        total_loss = recon_loss + 0.1 * privacy_loss
        
        # Log metrics
        self.log(f"{step_type}_recon_loss", recon_loss, prog_bar=True)
        self.log(f"{step_type}_privacy_loss", privacy_loss, prog_bar=True)
        self.log(f"{step_type}_total_loss", total_loss, prog_bar=True)
        
        return {'loss': total_loss, 'privatized': privatized, 'features': features}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

class DeepPrivacy2Privatizer(pl.LightningModule):
    def __init__(self, channels, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Encoder
        self.encoder = nn.ModuleList([
            ConvBlock3D(channels, 64),
            ConvBlock3D(64, 128),
            ConvBlock3D(128, 256),
            ConvBlock3D(256, 512)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder - Modified to handle correct channel dimensions
        self.decoder = nn.ModuleList([
            DeconvBlock3D(512, 256),    # Takes 512 from bottleneck + 512 from skip = 1024
            DeconvBlock3D(256, 128),    # Takes 256 from prev layer + 256 from skip = 512
            DeconvBlock3D(128, 64),     # Takes 128 from prev layer + 128 from skip = 256
            FinalDecoderBlock(64, channels)  # Takes 64 from prev layer + 64 from skip = 128
        ])
        
        # Identity branch
        self.identity = nn.Conv3d(channels, channels, 1)
        
        # Final output
        self.output = nn.Conv3d(channels * 2, channels, 1)

    def forward(self, x):
        # Store skip connections
        skips = []
        
        # Encoding
        out = x
        for encoder in self.encoder:
            out = encoder(out)
            skips.append(out)
        
        # Bottleneck
        out = self.bottleneck(out)
        
        # Decoding with skip connections
        skips = skips[::-1]  # Reverse skip connections
        for i, decoder in enumerate(self.decoder):
            skip = skips[i]
            # Concatenate skip connection
            out = torch.cat([out, skip], dim=1)
            out = decoder(out)
        
        # Identity branch
        identity = self.identity(x)
        
        # Combine privatized and identity features
        out = torch.cat([out, identity], dim=1)
        out = self.output(out)
        
        return out, identity

    def _common_step(self, batch, batch_idx, step_type):
        x, _ = batch
        privatized, features = self(x)
        
        # Calculate losses
        recon_loss = F.mse_loss(privatized, x)
        identity_loss = F.mse_loss(features, x)
        privacy_loss = torch.mean(torch.abs(privatized - x))
        
        total_loss = recon_loss + 0.1 * privacy_loss + 0.05 * identity_loss
        
        # Log metrics
        self.log(f"{step_type}_recon_loss", recon_loss, prog_bar=True)
        self.log(f"{step_type}_identity_loss", identity_loss, prog_bar=True)
        self.log(f"{step_type}_privacy_loss", privacy_loss, prog_bar=True)
        self.log(f"{step_type}_total_loss", total_loss, prog_bar=True)
        
        return {'loss': total_loss, 'privatized': privatized, 'features': features}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

# Helper modules for StyleGAN2
class ConstantInput(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channels, 1, 1))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, x.shape[2], x.shape[3])
        return out

class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim)
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        out = self.conv(x, style)
        out = self.activate(out)
        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])
        return x + self.weight * noise

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size//2)
        self.style = nn.Linear(style_dim, in_channel)

    def forward(self, x, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        x = x * style
        return self.conv(x)

# Helper modules for DeepPrivacy2
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class DeconvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # in_channels * 2 accounts for the skip connection concatenation
            nn.ConvTranspose3d(in_channels * 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class FinalDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    # Parametri di test
    batch_size = 1
    channels = 3
    depth = 20
    height = 256
    width = 256

    # Creazione degli input casuali
    input_data = torch.randn(batch_size, channels, depth, height, width)

    # Inizializzazione dei modelli
    stylegan2_model = StyleGAN2Privatizer(channels=channels)
    deepprivacy2_model = DeepPrivacy2Privatizer(channels=channels)

    # Passaggio in modalità di valutazione
    stylegan2_model.eval()
    deepprivacy2_model.eval()

    # Inferenza con StyleGAN2Privatizer
    with torch.no_grad():
        stylegan2_output, stylegan2_identity = stylegan2_model(input_data)
        print("Output StyleGAN2Privatizer:", stylegan2_output.shape)
        print("Identity StyleGAN2Privatizer:", stylegan2_identity.shape)

    # Inferenza con DeepPrivacy2Privatizer
    with torch.no_grad():
        deepprivacy2_output, deepprivacy2_identity = deepprivacy2_model(input_data)
        print("Output DeepPrivacy2Privatizer:", deepprivacy2_output.shape)
        print("Identity DeepPrivacy2Privatizer:", deepprivacy2_identity.shape)
