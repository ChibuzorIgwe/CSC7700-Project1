import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from mlp import MultilayerPerceptron, Layer, Relu, Softmax, CrossEntropy

# Load MNIST dataset from OpenML
print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255.0, mnist.target.astype(int)  # Normalize pixel values

# One-hot encode labels
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y.reshape(-1, 1))

# Split into 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP architecture
layers = [
    Layer(fan_in=784, fan_out=128, activation_function=Relu()),
    Layer(fan_in=128, fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=10, activation_function=Softmax())
]


mlp = MultilayerPerceptron(layers=layers)


# Train the model using RMSprop
train_losses, val_losses = mlp.train(
    X_train, y_train, X_val, y_val, 
    loss_func=CrossEntropy(), 
    learning_rate=0.001, 
    batch_size=64, 
    epochs=10, 
    rmsprop=True
)

# Plot the loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

# Evaluate the model on the test set
y_pred = mlp.forward(X_val)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1))
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Select one sample for each class (0-9) from the validation set
unique_classes = np.argmax(y_val, axis=1)
selected_indices = []
for digit in range(10):
    idx = np.where(unique_classes == digit)[0][0]
    selected_indices.append(idx)

selected_images = X_val[selected_indices].reshape(-1, 28, 28)
predicted_labels = np.argmax(y_pred[selected_indices], axis=1)

# Display the selected images with predicted labels
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(selected_images[i], cmap='gray')
    ax.set_title(f'Pred: {predicted_labels[i]}')
    ax.axis('off')
plt.show()
