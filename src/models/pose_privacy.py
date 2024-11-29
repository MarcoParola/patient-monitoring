import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePrivacyCNN(nn.Module):
    def __init__(self, input_shape=(3, 20, 256, 256), metadata_dim=3, num_classes=(5, 4, 3)):
        super(PosePrivacyCNN, self).__init__()

        # Video processing branch
        self.conv1 = nn.Conv3d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Adaptive pooling for video input
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # Flatten layer
        self.flatten = nn.Flatten()

        # Dropout layers for video branch
        self.dropout1 = nn.Dropout3d(p=0.1)
        self.dropout2 = nn.Dropout3d(p=0.1)
        self.dropout3 = nn.Dropout3d(p=0.1)
        self.dropout4 = nn.Dropout3d(p=0.1)

        # Metadata processing branch
        self.metadata_fc1 = nn.Linear(metadata_dim, 32)
        self.metadata_fc2 = nn.Linear(32, 64)

        # Dynamically calculate the flatten size for video input
        with torch.no_grad():
            self._calculate_flatten_size(input_shape)

        # Combined feature processing
        combined_feature_size = self.flattened_size + 64  # Video features + Metadata features

        # Fully connected layers
        self.fc1 = nn.Linear(combined_feature_size, 256)
        self.dropout_fc = nn.Dropout(p=0.4)

        # Output layers for each metadata task
        self.output_skin_color = nn.Linear(256, num_classes[0])  # Numero di classi per skin_color
        self.output_gender = nn.Linear(256, num_classes[1])      # Numero di classi per gender
        self.output_face = nn.Linear(256, num_classes[2])        # Numero di classi per face

        # Activation
        self.activation = nn.GELU()

    def _calculate_flatten_size(self, input_shape):
        # Create a dummy input to calculate the flattened size
        dummy_input = torch.zeros(1, *input_shape)
        x = self.conv1(dummy_input)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.dropout4(x)
        
        x = self.adaptive_pool(x)
        self.flattened_size = self.flatten(x).size(1)

    def forward(self, video_input, metadata_input):
        # Video processing branch
        x_video = self.activation(self.conv1(video_input))
        x_video = self.dropout1(x_video)
        
        x_video = self.activation(self.conv2(x_video))
        x_video = self.dropout2(x_video)
        
        x_video = self.activation(self.conv3(x_video))
        x_video = self.dropout3(x_video)
        
        x_video = self.activation(self.conv4(x_video))
        x_video = self.dropout4(x_video)
        
        # Adaptive pooling
        x_video = self.adaptive_pool(x_video)
        x_video = self.flatten(x_video)
        
        # Metadata processing branch
        x_metadata = self.activation(self.metadata_fc1(metadata_input))
        x_metadata = self.activation(self.metadata_fc2(x_metadata))
        
        # Combine video and metadata features
        x_combined = torch.cat([x_video, x_metadata], dim=1)
        
        # Fully connected layers
        x = self.activation(self.fc1(x_combined))
        x = self.dropout_fc(x)

        # Output for each metadata task
        output_skin_color = self.output_skin_color(x)
        output_gender = self.output_gender(x)
        output_face = self.output_face(x)

        return output_skin_color, output_gender, output_face

def test_model():
    # Video input shape: (batch_size, channels, depth, height, width)
    video_input_shape = (1, 3, 20, 256, 256)
    
    # Metadata input shape: (batch_size, metadata_features)
    metadata_input_shape = (1, 3)  # skin_color, gender, face
    
    # Create model
    model = PosePrivacyCNN(input_shape=video_input_shape[1:], metadata_dim=metadata_input_shape[1], num_classes=(3, 2, 5))
    
    # Create sample inputs
    video_tensor = torch.randn(*video_input_shape)
    metadata_tensor = torch.randn(*metadata_input_shape)
    
    # Pass inputs through model
    output_skin_color, output_gender, output_face = model(video_tensor, metadata_tensor)
    
    print("Output skin_color shape:", output_skin_color.shape)  # (batch_size, 3)
    print("Output gender shape:", output_gender.shape)          # (batch_size, 2)
    print("Output face shape:", output_face.shape)              # (batch_size, 5)

# Run the test
test_model()
