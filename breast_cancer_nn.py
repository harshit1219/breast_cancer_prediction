import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

# Random Seed
tf.random.set_seed(3)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data (1).csv")

# Remove extra column if present
if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

print(df.head())
print(df.shape)

# Features and Target
X = df.drop(["id", "diagnosis"], axis=1)
Y = df["diagnosis"]

# Convert labels
Y = Y.map({"M": 1, "B": 0})

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network
model = keras.Sequential([
    keras.layers.Dense(20, activation='relu', input_shape=(30,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=20,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, Y_test)

print("\nAccuracy :", accuracy)

# Accuracy Graph
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
#plt.show()

# Loss Graph
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])
#plt.show()

# ==========================
# Prediction
# ==========================

input_data = (
11.76,21.60,74.72,427.9,0.08637,
0.04966,0.01657,0.01115,0.1495,0.05888,
0.4062,1.21,2.635,28.47,0.005857,
0.009758,0.01168,0.007445,0.02406,0.001769,
12.98,25.72,82.98,516.5,0.1085,
0.08615,0.05523,0.03715,0.2433,0.06563
)

input_array = np.asarray(input_data).reshape(1, -1)

input_array = scaler.transform(input_array)

prediction = model.predict(input_array)

print("\nPrediction Probability :", prediction)

predicted_class = np.argmax(prediction)

print("Prediction Class :", predicted_class)

if predicted_class == 0:
    print("\nBenign Tumor")
else:
    print("\nMalignant Tumor")

print("\nProgram Finished Successfully")