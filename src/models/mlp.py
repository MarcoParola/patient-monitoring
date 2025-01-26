import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, task_type='binary_classification'):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.output = nn.Linear(128, output_dim)
        self.activation = nn.GELU()
        
        self.task_type = task_type
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.activation(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        output = self.output(x)
        
        # Activation based on task type
        if self.task_type == 'binary_classification':
            return torch.sigmoid(output)
        elif self.task_type == 'multiclass_classification':
            return torch.softmax(output, dim=1)
        elif self.task_type == 'regression':
            return output
    
if __name__ == "__main__":
    print("Test: PrivacyMLP")

    # Parametri di input (output del ConvBackbone)
    input_dim = 256  # La dimensione dell'output del ConvBackbone (flattened size)

    # Numero di classi per ciascun metadato
    output_dim_skin_color = 4  # Numero di classi per skin color
    output_dim_gender = 1  # Numero di classi per gender (regressione logistica)
    output_dim_age = 1  # Numero di classi per age (regressione)

    # Modelli MLP per ciascun metadato
    model_skin_color = MLP(input_dim, output_dim_skin_color, task_type='multiclass_classification')
    model_gender = MLP(input_dim, output_dim_gender, task_type='binary_classification')
    model_age = MLP(input_dim, output_dim_age, task_type='regression')

    # Creiamo l'input (output del ConvBackbone)
    x_skin_color = torch.randn(8, input_dim)  # Output del ConvBackbone
    output_skin_color = model_skin_color(x_skin_color)
    print(f"Skin color output shape: {output_skin_color.shape}")  # Dovrebbe essere (batch_size, 4)

    x_gender = torch.randn(8, input_dim)  # Output del ConvBackbone
    output_gender = model_gender(x_gender)
    print(f"Gender output shape: {output_gender.shape}")  # Dovrebbe essere (batch_size, 1)

    x_age = torch.randn(8, input_dim)  # Output del ConvBackbone
    output_age = model_age(x_age)
    print(f"Age output shape: {output_age.shape}")  # Dovrebbe essere (batch_size, 1)

    print("\nTest MLP completato con successo!")