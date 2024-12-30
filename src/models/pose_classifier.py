import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from src.models.conv_backbone import CNN3DLightning
from src.models.skeleton.yolo_skeleton import YOLOPoseAPI
from src.models.skeleton.openpose.openpose_skeleton import OpenPoseAPI
from src.models.mlp import MLP
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .model_util import label_to_one_hot

class PoseClassifier(pl.LightningModule):
    def __init__(self, input_shape=(1, 3, 20, 640, 480), output_dim=9, backbone="CNN3D", freeze=False, detect_face=False, detect_hands=False, fps = 30, model_size="n"):
        super(PoseClassifier, self).__init__()
        self.conv_backbone = None
        if (backbone == "YOLO"):
            self.feature_dim = 17 * 2  # Numero di keypoints per YOLO
        if (backbone == "CNN3D"):
            self.conv_backbone = CNN3DLightning(in_channels=input_shape[1])
        elif (backbone == "OpenPose"):
            self.conv_backbone = OpenPoseAPI(detect_face=detect_face, detect_hands=detect_hands, fps=fps)
            self.feature_dim = 18  # Numero base di keypoints per OpenPose senza viso né mani
            if detect_face:
                self.feature_dim += 70  # Aggiungi 70 keypoints per il viso
            if detect_hands:
                self.feature_dim += 42  # Aggiungi 21 keypoints per ogni mano (2 mani)

            self.feature_dim = 75  # Ogni keypoint ha 3 coordinate (x, y, confidence), date per ogni frame, per 5 secondi

       
        # MLP per classificazione finale
        self.mlp = MLP(input_dim=self.feature_dim, output_dim=output_dim)
        self.test_outputs = [] 
        self.output_dim = output_dim
        

    def forward(self, video_input):
        if (self.conv_backbone.__class__.__name__ == "CNN3DLightning"):
            video_input = self.conv_backbone(video_input)
            video_input = video_input.flatten()
    
        output = self.mlp(video_input)
        return output


    def _common_step(self, batch, step_type):
        video_input, labels = batch
        
        # Predizione
        pred = self(video_input)
        loss = F.cross_entropy(pred, labels)

        # Logging delle metriche con WandB tramite self.log
        self.log(f"{step_type}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step_type}_accuracy", self.compute_accuracy(pred, labels), prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'logits': pred}
    
    
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, step_type="val")

    def test_step(self, batch, batch_idx):
        results = self._common_step(batch, step_type="test")
        self.test_outputs.append({
            "logits": results["logits"],
            "labels": batch[1]
        })  # Aggiungi logits e label alla lista
        return results

    def configure_optimizers(self):
        # Configurazione dell'ottimizzatore
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def on_train_epoch_end(self):
        print("\n** on_train_epoch_end **")

    def on_validation_epoch_end(self):
        print("\n** on_validation_epoch_end **")

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("No test outputs collected.")
            return

        # Raccolta di tutte le predizioni e label
        all_preds = torch.cat([torch.argmax(out["logits"], dim=1) for out in self.test_outputs], dim=0)
        all_labels = torch.cat([out["labels"] for out in self.test_outputs], dim=0)

        # Calcolo della matrice di confusione
        conf_matrix = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy(), labels=list(range(self.output_dim)))

        # Visualizzazione e logging della matrice di confusione
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(self.output_dim)), yticklabels=list(range(self.output_dim)))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        self.logger.experiment.log({"Confusion Matrix": plt.gcf()})
        plt.close()

        # Svuota la lista degli output per il prossimo test
        self.test_outputs.clear()

    def compute_accuracy(self, preds, labels):
        # Calcolo dell'accuratezza
        _, predicted = torch.max(preds, 1)
        correct = (predicted == labels).float().sum()
        accuracy = correct / labels.size(0)
        return accuracy

if __name__ == "__main__":
    print("Test: PoseClassifier")

    # Configurazione
    input_shape = (5, 3, 20, 256, 256)  # Batch size di 5
    num_classes = 9  # Numero di classi unificate
    model = PoseClassifier(input_shape=input_shape, output_dim=num_classes)

    # Generazione di input casuali
    video_input = torch.randn(input_shape)  # Input video
    labels = torch.randint(0, num_classes, (input_shape[0],))  # Classi unificate
    batch = (video_input, labels)

    # Test del metodo forward
    output = model(video_input)
    print(f"Output shape: {output.shape}")

    # Test del training_step
    train_loss = model.training_step(batch, batch_idx=0)
    print(f"Training loss: {train_loss['loss'].item()}")

    # Test del validation_step
    val_loss = model.validation_step(batch, batch_idx=0)
    print(f"Validation loss: {val_loss['loss'].item()}")

    # Test del test_step
    test_loss = model.test_step(batch, batch_idx=0)
    print(f"Test loss: {test_loss['loss'].item()}")

    print("\nTest completato con successo!")

