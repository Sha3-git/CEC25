# Tumor Detection Model

This project is a deep learning-based tumor detection system using PyTorch and EfficientNetV2. It trains a model to classify brain MRI images as either containing a tumor ("yes") or not ("no").

## Prerequisites

Before running this project, you need to have the following installed:

- Python 3.9.21
- PyTorch 2.1.0
- torchvision 0.16.0
- Pillow 10.0.0
- numpy 1.24.0
- Other dependencies listed in requirements.txt

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Environment Setup

This project requires setting an environment variable to locate your dataset:

```bash
# On Windows
set CEC_2025_dataset=path\to\your\dataset

# On macOS/Linux
export CEC_2025_dataset=/path/to/your/dataset
```

The script will use this environment variable to find your dataset directory.

## Dataset Structure

The dataset should be organized as follows:

```
CEC_2025_dataset/
└── CEC_2025/
    ├── yes/
    │   ├── yes__001.png
    │   ├── yes__002.png
    │   └── ...
    ├── no/
    │   ├── no__001.png
    │   ├── no__002.png
    │   └── ...
    └── CEC_test/
        └── test_001.png
        └── test_002.png
        └── ...
```

- The `yes` folder contains MRI images with tumors
- The `no` folder contains MRI images without tumors
- The `CEC_test` folder contains test images for prediction

## Workflow Sequence Diagram

The following sequence diagram illustrates the workflow of the tumor detection system:

```mermaid
sequenceDiagram
    participant User
    participant run.py
    participant Model
    participant Dataset

    User->>User: Set CEC_2025_dataset environment variable
    User->>run.py: Execute run.py
    run.py->>Model: Load trained model (final_model.pth)
    run.py->>Dataset: Load test images from CEC_test folder
    loop For each test image
        run.py->>Model: Process and predict image
        Model-->>run.py: Return prediction & confidence score
        run.py->>run.py: Classify probability (Very Unlikely to Very Likely)
        run.py->>run.py: Add result to output data
    end
    run.py->>User: Save results to output.csv
    run.py->>User: Display average confidence score
```

## Running the Model

To run the model on the CEC_test dataset (after setting environment var.):

```bash
python run.py
```

The run script will:
1. Load the trained model from `final_model.pth` file
2. Process all images in the `CEC_test` folder
3. Generate predictions with confidence scores
4. Save results to `output.csv` in same directory as script

## Additional Scripts

<details>
<summary><b>Testing the Model (test.py)</b></summary>

To test the model's performance on the CEC_test dataset:

```bash
python test.py
```

The test script will:
1. Load the trained model from `final_model.pth` file
2. Process all images in the `CEC_test` folder
3. Generate predictions with confidence scores
4. Save results to `output.csv`

You can modify `NUM_IMAGES` in `test.py` to change the number of test images (default: 50).
</details>

<details>
<summary><b>Training the Model (train.py)</b></summary>

To train the model, run:

```bash
python train.py
```

The training script will:
1. Load and preprocess the images from the `yes` and `no` folders
2. Train an EfficientNetV2 model on these images
3. Save the trained model as `tumor_model_1000.pth`

You can modify the following parameters in `train.py`:
- `NUM_IMAGES`: Number of images to use for training
- `MODEL_NAME`: Name of the saved model file
- `epochs`: Number of training epochs
</details>

## Understanding the Results

After running the script, you will see:
- A CSV file (`output.csv`) containing:
  - File name
  - Prediction (Yes/No)
  - Confidence score (0-1)
  - Probability classification (Very Unlikely, Unlikely, Likely, Very Likely)
- Average confidence score across all tested images
- Total number of images tested

The confidence score interpretation:
- < 0.5: Very Unlikely
- 0.5-0.75: Unlikely
- 0.75-0.9: Likely
- > 0.9: Very Likely

## Troubleshooting

If you encounter PyTorch compatibility issues, make sure you have Python 3.9 and PyTorch 2.1.0 installed. For newer Python versions, you may need to use the latest pre-release version of PyTorch:

```bash
pip install --pre torch torchvision torchaudio
```

If the scripts cannot find the dataset, verify that:
1. The environment variable `CEC_2025_dataset` is correctly set
2. Your folder structure matches the one described above
3. The images in the CEC_test folder are in a supported format (png, jpg, jpeg) 