# patient-monitoring

This github repo for the patient monitoring project between the [University of Pisa](https://www.unipi.it/), the University of [Clermont-Auvergne](https://www.uca.fr/), and [Smarty](https://sma-rty.com/).

More information in the [official documentation](docs/README.md).

# Setup Instructions for Patient Monitoring Project

## Step 1: Set Up Conda Environment
1. Open a terminal.
2. Create a new conda environment with Python version 3.11.10 and name it `patient-monitoring`:
   ```bash
   conda create --name patient-monitoring python=3.11.10 -y
   ```
3. Activate the newly created environment:
   ```bash
   conda activate patient-monitoring
   ```
4. Install the required Python packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Step 2: Download Files
Download the following files [here](https://drive.google.com/drive/folders/15gRDlVj5_ZLkJXTS8QrZmNXgMyzZVIkv?usp=sharing):
- `data.zip`  
- `dataset.zip`

## Step 3: Extract Files
Extract the downloaded files into the root directory of the `patient-monitoring` project.

## Step 4: Configure `wandb.yaml`
1. Navigate to the `config` folder.  
2. Create or edit the file `wandb.yaml`.  
3. Add the following content:  

   ```yaml
   wandb:
     entity: "baffobello14-universit-di-pisa"
     project: "patient-monitoring"
   ```
