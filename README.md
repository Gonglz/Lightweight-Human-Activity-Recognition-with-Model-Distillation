# Lightweight Human Activity Recognition with Model Distillation

This project focuses on human activity recognition using time-series data from wearable sensors. By leveraging teacher-student knowledge distillation, the aim is to improve model efficiency while maintaining accuracy for deployment on resource-constrained edge devices.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project implements knowledge distillation to build a lightweight student model for human activity recognition (HAR). The teacher model is first trained on the full dataset, and the student model is trained using the soft labels provided by the teacher. This approach reduces computational overhead, making it suitable for edge devices.

## Dataset

We use the following datasets for HAR:

1. [PAMAP2 Physical Activity Monitoring Dataset](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
2. [MHEALTH Dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)

The datasets are preprocessed to align features and activities before training and evaluation.

## Project Structure

```
Project/
├── dataset/
│   ├── PAMAP2/
│   │   ├── Optional/
│   │   ├── Protocol/
│   │   └── Processed/
│   ├── MHEALTH/
│   │   └── clean/
│   ├── combined_har_data.csv
│   └── preprocessed_train_test.npz
├── output/
│   ├── activity_recognition_model_sgd.h5
│   ├── confusion_matrix.png
│   ├── evaluation_output.txt
│   ├── teacher_soft_labels.npy
│   └── training_validation_curves.png
├── output_offline/
├── output_online/
├── MHEALTH_dp.py
├── PAMAP_dp.py
├── data_combine.py
├── offline_distill.py
├── online_distill.py
├── train.py
├── requirements.txt
└── .gitignore
```

## Environment

### Python Version
This project is developed using **Python 3.8**. Ensure you have the correct version installed.

### Required Packages

- `imblearn==0.7.0`
- `matplotlib==3.5.3`
- `numpy==1.19.5`
- `pandas==1.3.5`
- `scikit_learn==1.0.2`
- `scipy==1.7.3`
- `seaborn==0.12.2`
- `tensorflow==2.4.0`
- `tqdm==4.64.1`

### Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install imblearn==0.7.0 matplotlib==3.5.3 numpy==1.19.5 pandas==1.3.5 scikit_learn==1.0.2 scipy==1.7.3 seaborn==0.12.2 tensorflow==2.4.0 tqdm==4.64.1
```



## Usage

### Preprocessing
Run the preprocessing scripts to clean and combine datasets:

```bash
python MHEALTH_dp.py
python PAMAP_dp.py
python data_combine.py
```

### Training
Train the teacher and student models using:

```bash
python train.py
python offline_distill.py
python online_distill.py
```

### Evaluation
Evaluation results and visualizations can be found in the `output`, `output_offline`, and `output_online` directories.

The outputs in output, _online, _offline are all of 10-epoches for making a fast demo. 
To make a good performance, 30-epoches is a good choice.

## Acknowledgments

We acknowledge the contributors of the [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) and [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) datasets. These datasets were critical for the success of this project.
