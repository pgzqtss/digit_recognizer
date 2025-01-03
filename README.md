# Digit Recognizer

This project is a digit recognition system using a Convolutional Neural Network (CNN) implemented with Keras and TensorFlow. The model is trained on the MNIST dataset and can predict handwritten digits from 0 to 9.

## Files

- `digit_recognizer.py`: Main script for training the CNN model and generating predictions.
- `check.py`: Script to validate the predictions by comparing them with the expected labels.
- `explain.txt`: Documentation explaining the project and the neural network layers.
- `model.h5`: Saved trained model.
- `out.csv`: Output file containing the predicted labels.
- `requirements.txt`: List of dependencies required to run the project.
- `sample_submission.csv`: Sample submission file.
- `test.csv`: Test dataset.
- `train.csv`: Training dataset.

## Setup

1. Clone the repository:
    ```sh
    git clone <https://github.com/pgzqtss/digit_recognizer>
    cd <https://github.com/pgzqtss/digit_recognizer>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Train the model and generate predictions:
    ```sh
    python digit_recognizer.py
    ```

2. Validate the predictions:
    ```sh
    python check.py
    ```

## Explanation

### Data Preparation

- The training data is loaded from [train.csv] and the first 5000 rows are used for training.
- The data is split into features and labels.
- The features are normalized and resized to 128x128 to match the input requirements of the InceptionV3 model.

### Model Architecture

- The model is a Convolutional Neural Network (CNN) with multiple convolutional and max-pooling layers.
- The final layers are dense layers for decision making and classification.
- The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

### Training

- The model is trained for 10 epochs on the normalized training data.
- The trained model is saved as [model.h5].

### Prediction

- The test data is loaded from [test.csv] and normalized.
- The trained model is used to predict the labels for the test data.
- The predictions are saved in [out.csv].

### Validation

- The [check.py] script loads the handwritten images and the predicted output. Allow us to compare whether it is correct or not.
