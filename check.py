import pandas as pd
import numpy as np
import sys
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Set the encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the pre-trained model
model = load_model('model.h5')

# Load the CSV file
df = pd.read_csv("out.csv")

# Load the test data
test = pd.read_csv('test.csv')
X_test = test / 255.0

# Predict the labels
p_data = model.predict(X_test)

# Generate the expected labels
expected_labels = [np.argmax(item) for item in p_data]

# Compare the labels and display images
correct = True
for i, label in enumerate(expected_labels, start=1):
    actual_label = df.loc[df['ImageId'] == i, 'Label'].values[0]
    if actual_label != label:
        correct = False
        print(f"Mismatch at Image Id {i}: expected {label}, got {actual_label}")
        
        # Display the image
        plt.imshow(X_test.iloc[i-1].values.reshape(28, 28), cmap='gray')
        plt.title(f"Image Id {i}: expected {label}, got {actual_label}")
        plt.show()
    else:
        # Display the image even if it matches
        plt.imshow(X_test.iloc[i-1].values.reshape(28, 28), cmap='gray')
        plt.title(f"ImageId {i}: correct label {actual_label}")
        plt.show()

if correct:
    print("The CSV file is correct.")
else:
    print("There are mismatches in the CSV file.")