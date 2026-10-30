# Plant Disease Detection Using Deep Learning

A comparative deep-learning practical that explores three approaches for distinguishing healthy and diseased plant-leaf images: a custom Convolutional Neural Network (CNN), ResNet50 transfer learning, and a convolutional autoencoder for anomaly detection.

> **Important dataset note:** The notebook describes the PlantVillage dataset and provides a Kaggle reference, but the code that was executed does **not** download or load the real PlantVillage images. It generates a balanced synthetic dataset of leaf-like images (green images for the Healthy class and images with artificial brown/yellow spots for the Diseased class). All reported results below are from this synthetic dataset and should not be interpreted as real-world plant-disease detection performance.

## Project Overview

Early identification of plant disease can help reduce crop losses. This practical compares supervised image classification with an unsupervised reconstruction-based anomaly-detection approach.

The task is binary classification:

| Label | Meaning |
|---|---|
| `0` | Healthy |
| `1` | Diseased |

### Implemented approaches

1. **Custom CNN:** A convolutional neural network built from scratch, with convolution, batch-normalization, pooling, dropout, and dense classification layers.
2. **ResNet50 transfer learning:** An ImageNet-pretrained ResNet50 backbone is used without its original classification head. The notebook first freezes the backbone to train a custom classification head, then fine-tunes the final layers.
3. **Convolutional autoencoder:** The autoencoder is trained only on Healthy training images. At inference time, reconstruction error is used as an anomaly score; images with error above a threshold are marked as Diseased.

## Dataset and Preprocessing

### Current notebook implementation

The notebook creates the data programmatically using a synthetic image generator:

- **Total images:** 4,000
- **Healthy images:** 2,000
- **Diseased images:** 2,000
- **Image dimensions:** 128 × 128 pixels
- **Channels:** 3 (RGB)
- **Classes:** 2 (Healthy and Diseased)

Healthy samples are generated with green-dominant colors, texture noise, and simulated leaf veins. Diseased samples use the same base generation with artificial brown, yellow, and dark-red spots.

The dataset is split using stratified train/validation/test splitting:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 2,800 | 70% |
| Validation | 600 | 15% |
| Test | 600 | 15% |
| **Total** | **4,000** | **100%** |

Training images are augmented with random rotation, horizontal flipping, width/height shifts, zoom, and brightness changes.

### PlantVillage reference

The notebook mentions PlantVillage as the intended public dataset and includes this reference:

[Kaggle – Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

This dataset is **not loaded by the current notebook implementation**. To conduct a real-data experiment, replace the synthetic image-generation section with code that loads and preprocesses the downloaded PlantVillage images, and then retrain and reevaluate all models.

## Model Details

### 1. Custom CNN

- Four convolutional feature-extraction blocks.
- Filter sizes progress through 32, 64, 128, and 256.
- Uses ReLU activations, batch normalization, max pooling, and dropout.
- Global average pooling followed by dense layers (256 and 128 units).
- Two-unit softmax output for Healthy and Diseased classes.
- **Parameters:** 684,962 (as reported by the notebook).

### 2. ResNet50 Transfer Learning

- ResNet50 initialized with ImageNet weights and `include_top=False`.
- The backbone is initially frozen while a custom classification head is trained.
- The final 20 backbone layers are then unfrozen for fine-tuning with a smaller learning rate.
- Custom head uses global average pooling, dense layers, batch normalization, dropout, and a two-class softmax output.
- **Parameters:** 24,770,690 (as reported by the notebook).

### 3. Convolutional Autoencoder

- Encoder compresses each 128 × 128 RGB image into a latent representation of shape **8 × 8 × 64**.
- Decoder reconstructs the image from this compressed representation.
- Trained using Healthy images only, with the input image also used as the reconstruction target.
- Mean squared error (MSE) between the input and reconstructed image is used as the anomaly score.
- The notebook sets the anomaly threshold to the mean Healthy reconstruction error plus two standard deviations.
- **Parameters:** 373,251 (as reported by the notebook).

## Results Recorded in the Notebook

The following are the saved test results from the notebook's synthetic dataset run:

| Model | Approach | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score | AUC-ROC |
|---|---|---:|---:|---:|---:|---:|
| Custom CNN | Supervised | 50.00% | 25.00% | 50.00% | 0.3333 | 0.5000 |
| ResNet50 | Supervised transfer learning | 50.00% | 25.00% | 50.00% | 0.3333 | 0.2246 |
| Convolutional Autoencoder | Unsupervised anomaly detection | 97.17% | 97.28% | 97.17% | 0.9716 | 0.9993 |

### How to interpret these results

- In the recorded run, both the Custom CNN and ResNet50 predicted every test image as Diseased. Consequently, they achieved 50% accuracy on the balanced test set but did not correctly identify the Healthy class.
- The autoencoder achieved 97.17% accuracy on the synthetic test images. The notebook reports a Healthy reconstruction-error mean of `0.006369`, a Diseased mean of `0.049744`, and a threshold of `0.012669`.
- The autoencoder threshold is calculated using reconstruction errors from the Healthy **test** images in the notebook. For a rigorous evaluation, set the threshold using a separate validation set of Healthy images and keep the test set untouched until final evaluation.
- These results measure separation of the notebook's artificially generated image patterns. They do not establish accuracy on photographs of actual plant diseases.

## Visualizations and Analysis

When the notebook is executed, it creates visual outputs for:

- Sample Healthy and Diseased synthetic images.
- Class distribution and data augmentation examples.
- CNN and ResNet50 training histories and confusion matrices.
- Autoencoder training history, reconstruction examples, and reconstruction-error distributions.
- Model metric comparison, radar chart, and ROC curves.
- Grad-CAM heatmaps for CNN feature-activation interpretation.
- t-SNE visualization of the autoencoder's latent representations.

The notebook saves these visualizations as PNG files in the current working directory.

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow, Keras
- **Numerical/data processing:** NumPy, Pandas
- **Image processing:** OpenCV, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn

## Installation

Use Python in a Jupyter Notebook environment or Google Colab. Install the libraries used by the notebook:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow kaggle
```

ResNet50 loads pretrained ImageNet weights, so an internet connection may be needed the first time that part of the notebook runs.

## How to Run

1. Clone or download this repository.
2. Open `Plant_Disease_Detection_DL_Practical_10(20261030-091121).ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells from top to bottom.
4. The notebook installs/imports dependencies, generates the synthetic dataset, preprocesses the images, trains the three models, and displays the evaluation results and visualizations.

## Suggested Repository Structure

```text
Plant-Disease-Detection/
│
├── README.md
├── Plant_Disease_Detection_DL_Practical_10(20261030-091121).ipynb
└── outputs/                 # Optional: add generated plots here
```

## Limitations and Future Improvements

- Replace the synthetic image generator with the actual PlantVillage images (or another properly documented real-world plant-leaf dataset).
- Re-train all three approaches on real images and evaluate them using a held-out test set.
- Select the autoencoder's anomaly threshold using validation data rather than test data.
- Investigate why the supervised classifiers predict only one class; review data quality, preprocessing, class signals, training behavior, and model configuration.
- Evaluate performance across individual plant species and disease categories rather than only Healthy versus Diseased.
- Test with field photographs that include natural lighting, backgrounds, occlusion, and varied image quality.

## Disclaimer

This repository is an educational deep-learning practical and experimental comparison. It is not a validated agricultural diagnostic tool. Do not use its current synthetic-data results to make crop-treatment or agricultural decisions.
