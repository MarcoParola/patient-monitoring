# patient-monitoring

This github repo for the patient monitoring project between the [University of Pisa](https://www.unipi.it/), the University of [Clermont-Auvergne](https://www.uca.fr/), and [Smarty](https://sma-rty.com/).

More information in the [official documentation](docs/README.md).

# Setup Instructions for Patient Monitoring Project

## Step 1: Download Files  
Download the following files [here](https://drive.google.com/drive/folders/15gRDlVj5_ZLkJXTS8QrZmNXgMyzZVIkv?usp=sharing): 
- `data.zip`  
- `dataset.zip`

## Step 2: Extract Files  
Extract the downloaded files into the root directory of the `patient-monitoring` project.

## Step 3: Configure `wandb.yaml`  
1. Navigate to the `config` folder.  
2. Create or edit the file `wandb.yaml`.  
3. Add the following content:  

   ```yaml
   wandb:
     entity: "baffobello14-universit-di-pisa"
     project: "patient-monitoring"

