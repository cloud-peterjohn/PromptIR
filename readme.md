# Image Restoration using PromptIR + SE Module

## Introduction

This project aims to restore images degraded by rain and snow using a single model. We utilize the PromptIR model combined with an improved SE (Squeeze-and-Excitation) module to enhance restoration performance.

## Dataset Preparation

1.  **Download the Dataset**:
    Please download the dataset from the following link:
    [https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul?usp=share_link](https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul?usp=share_link)

2.  **Place the Dataset**:
    After downloading, extract and place the dataset in the project's root directory. Ensure the file structure is as follows:

    ```
    <project_root>/
    ├── hw4_realse_dataset/
    │   ├── train/      # Training set data
    │   └── test/       # Test set data
    ├── ... (other project files)
    ```

## Environment Setup and Installation

1.  **Create a Conda Virtual Environment**:
    It is recommended to create a new Python 3.12 virtual environment using Conda:
    ```bash
    conda create -n promptir_env python=3.12
    conda activate promptir_env
    ```

2.  **Install Dependencies**:
    After activating the virtual environment, install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running

### Training the Model
```bash
python main.py
```
Training parameters can be adjusted in the `main.py` file. Trained model weights will be saved in the `checkpoints` directory.

### Testing the Model and Visualization
```bash
python test.py
```
Please ensure that you update the `model_path` in `test.py` to your trained model's path. Test results will be saved in `results/pred.npz`, and the visualization image will be saved in `results/visualization.svg`.

## Experimental Results

This project achieved a PSNR of **30.69** on the test set.


Ping-Yeh Chou 113550901
