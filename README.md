# Pixel Statistical Classifier

A lightweight image classification algorithm that performs classification using per-pixel statistical distributions rather than neural networks or traditional machine-learning libraries.

## Motivation
I wanted to investigate how much image-classification performance could be obtained from a deliberately simple statistical, pixel-based approach. I implemented the classifier from scratch, progressively experimented with different representations and techniques, and evaluated their effects. The final implementation achieved an overall accuracy of 82.16%, overall precision of 82.32% and overall recall of 81.85% on the MNIST dataset.

The goal was not to compete with modern neural-network architectures, but to investigate the relationship between model simplicity, training data, and classification performance.

## How it works

### Training

During training, each grayscale image is converted to floating-point values and normalized to the `[0, 1]` range. The training images are then grouped by class, and the mean brightness of each pixel is calculated across all images belonging to the same class. This produces a spatial brightness map for each class, where each value represents the average brightness observed at that pixel position.

Each class brightness map is then independently **L2-normalized**. This scales every class representation to have a unit L2 norm, removing differences in overall magnitude while preserving the relative spatial brightness distribution of the class.

### Prediction

For prediction, the input image is first normalized to the `[0, 1]` range and then L2-normalized in the same way as the learned class brightness maps. Because both the input image and class maps have unit L2 norms, their dot product is equivalent to **cosine similarity**.

The normalized input image is compared against every normalized class brightness map, producing a similarity score for each class. The class whose brightness map has the highest cosine similarity to the input image is selected as the prediction.

## Experiments

The development of the classifier was performed incrementally through a series of Jupyter notebooks and models. Each experiment investigates a specific modification or question and records the resulting performance and analysis.
The MNIST dataset was used for evaluation. In the final experiment, NIST datasets: Fashion MNIST, KMNIST and EMNIST were also trained and tested on.
Results and analysis for each model can be found in its respective experiment notebook.
Info on each model can be found in the respective model file.

## Final Results (MNIST DATASET)

### Metrics
| Class | Accuracy | Precision | Recall | F1 Score |
|------:|---------:|----------:|-------:|---------:|
| 0 | 97.89% | 87.08% | 92.14% | 89.54% |
| 1 | 97.62% | 86.37% | 93.83% | 89.95% |
| 2 | 96.37% | 86.80% | 76.45% | 81.30% |
| 3 | 95.49% | 74.45% | 84.26% | 79.05% |
| 4 | 96.37% | 82.07% | 80.65% | 81.36% |
| 5 | 95.49% | 81.19% | 64.35% | 71.79% |
| 6 | 97.50% | 86.05% | 88.20% | 87.11% |
| 7 | 97.43% | 91.59% | 82.59% | 86.85% |
| 8 | 94.82% | 72.18% | 76.18% | 74.13% |
| 9 | 95.34% | 75.40% | 79.88% | 77.57% |
| **Overall** | **82.16%** | **82.32%** | **81.85%** | **81.87%** |

### Model Efficiency

| Metric | Value |
|---|---|
| CPU | Intel Core i3 10th Gen |
| GPU | Not used |
| Training set | 60,000 images |
| Test set | 10,000 images |
| Training time | 0.50-1.00 seconds |
| Prediction time | 0.50-1.00 seconds |

## Key findings

### High data efficiency, low performance ceiling

The final model reaches most of its eventual performance with relatively few training samples, as seen in the final experiment, where performance plateaus at approximately 2000 training samples on the EMNIST digits dataset.

Increasing the training set beyond this point produces little or no improvement. This suggests that the primary limitation is the representational capacity of the pixel-statistical approach rather than insufficient training data.

### Low resource usage

The classifier requires relatively few computational resources compared to modern machine-learning approaches. Its model consists primarily of statistical information derived directly from the training images, without the need for a large number of learned parameters or specialized hardware.

Training and prediction can therefore be performed using standard CPU-based computation and relatively small amounts of memory. This makes the approach suitable for environments where computational resources are limited.

### Fast training and prediction

The simplicity of the model also results in fast training and prediction. Training primarily consists of accumulating and processing pixel statistics for each class, while prediction involves comparing an input image against the learned class representations.

Unlike iterative optimization-based models, the classifier does not require backpropagation, gradient descent, or repeated parameter updates. This allows the model to be trained and evaluated quickly, particularly on smaller datasets.

### Weak predictive power compared to state-of-the-art models

Despite its data efficiency and low computational requirements, the classifier has substantially lower predictive performance than modern image-classification methods.

Its reliance on pixel-level statistics limits its ability to capture complex spatial relationships and higher-level visual features. Consequently, increasing the amount of training data provides little improvement once the statistical representation has stabilized.

The final accuracy should therefore not be interpreted as competitive with state-of-the-art image-classification models. Instead, the results demonstrate the amount of predictive information that can be extracted from a comparatively simple statistical representation, while also illustrating its fundamental limitations.

## Using the project

Clone the repository into your local workspace:
```
git clone https://github.com/Hamaz-Hamza/Pixel-Statistical-Classifier.git
```
Then:
```
cd Pixel-Statistical-Classifier

python3 -m venv .venv
source .venv/bin/activate

pip install .
```

`src/psc/models` contains the reusable classifier implementations, while `experiments/` contains the notebooks used to evaluate and analyze different versions of the approach. `src/psc/evaluation` contains python functions that can be used to evaluate the pixel statistical classifier models.

The code used in the notebooks will automatically download all used datasets into the `datasets/` folder except for the EMNIST dataset which must be downloaded manually. After downloading the .zip file, extract contents into `datasets/emnist` and then run the corresponding code.

The models use the standard `.fit`, `.predict`, `.predict_single` member functions for training, batch prediction, and single sample predictions.
```
model = PixelStatisticalClassifier(...)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
```
Inputs are expected to be numpy arrays. Images must have shape `(n, x, y)` and labels must have shape `(n,)`

Dependencies are listed in `pyproject.toml`, however if you installed the project using `pip install .`, then installing dependencies seperately is not needed
