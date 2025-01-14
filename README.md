# patient-monitoring

This github repo for the patient monitoring project between the [University of Pisa](https://www.unipi.it/), the University of [Clermont-Auvergne](https://www.uca.fr/), and [Smarty](https://sma-rty.com/).

More information in the [official documentation](docs/README.md).

# TODO

- parametrize `./scripts/preprocessing.py`
- write the documentation (all of us)
- implement `load_datasets(...)` in `src.datasets.dataset_utils` and `load_models(...)` in `src.models.model_utils`



# CATE - Training Experiments
Download Openpose models from [My Drive](https://drive.google.com/drive/folders/14oba9QaCp1bcvDVtLpfqHLpuCUI8_od_?usp=sharing)

Download Openpose release 1.6.0 form this [link](OPENPOSE_BINARIES_URL="https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.6.0/openpose-1.6.0-binaries-win64-only_cpu-flir-3d.zip"
)
unzip the file and then place it inside src/models/skeleton

-   remove the duplicate top folder 
-   move the downloaded models respectively into
    -   pose_iter_102000.caffemodel into models/hand 
    -   pose_iter_584000.caffemodel into models/pose/body_25
    -   pose_iter_440000.cafffemodel into models/pose/coco
    -   pose_iter_116000.caffemodel into models/face
-   create a folder output/output_jsons inside openpose main folder
-   in skeleton fonder move openpose_skeleton.py inside the openpose main folder 
In bash_cate.sh line 21 modify the path to the path to your anaconda venv

execute bash_cate.sh

Training logs are at [My WandB](https://wandb.ai/c-bruchi-university-of-pisa/pat-mon/overview), containing losses, confusion matrixes and the final models.
In dataset/keypoints are the extracted keypoints csv for YOLO and Openpose Runs

The experiments are tested at 1,2,3 FPS, and with resolution 640x480 for compliance with training datasets for YOLO (in all models sizes) and OpenPose with all keypoints combination available (face, body, hands), and 256x256 for the CNN. 



