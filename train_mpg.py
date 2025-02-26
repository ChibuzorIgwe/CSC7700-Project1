import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlp import MultilayerPerceptron, Layer, SquaredError, Relu, Linear, SquaredError

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
data = pd.read_csv(dataset_url, delim_whitespace=True, names=columns + ["car name"], na_values="?")

# Drop unnecessary column and handle missing values
data.drop(columns=["car name"], inplace=True)
data.dropna(inplace=True)

# Split the data into features and target
X = data.drop('mpg', axis=1).values
y = data['mpg'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Define the MLP architecture
layers = [
    Layer(X_train.shape[1], 64, Relu(), dropout_rate=0.2),
    Layer(64, 32, Relu(), dropout_rate=0.2),
    Layer(32, 1, Linear())
]

# Instantiate the MLP
mlp = MultilayerPerceptron(layers=layers)

# Training parameters
epochs = 10000  # Adjusted for efficiency
learning_rate = 0.001
batch_size = 32

# Train the model
train_losses, val_losses = mlp.train(X_train, y_train, X_val, y_val, SquaredError(), learning_rate, batch_size, epochs, rmsprop=True)

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 20)
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Compute total testing loss
y_test_pred = mlp.forward(X_test)
test_loss = SquaredError().loss(y_test, y_test_pred)
print(f"Total Testing Loss: {test_loss:.4f}")

# Select 10 random test samples and compare predicted vs. true MPG
sample_indices = np.random.choice(len(X_test), 10, replace=False)
sample_X = X_test[sample_indices]
sample_y_true = y_test[sample_indices]
sample_y_pred = mlp.forward(sample_X)

# Create a DataFrame for visualization
sample_results = pd.DataFrame({
    "True MPG": sample_y_true.flatten(),
    "Predicted MPG": sample_y_pred.flatten()
})

# Print the table in a readable format
print("\nPredicted vs True MPG:")
print(sample_results.to_string(index=False))
