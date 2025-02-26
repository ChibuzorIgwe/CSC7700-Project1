import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    n_samples = len(train_x)
    for i in range(0, n_samples, batch_size):
        batch_x = train_x[i:i + batch_size]
        batch_y = train_y[i:i + batch_size] 
        yield batch_x, batch_y
    pass


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)
    pass


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    pass


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    pass


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)
    pass


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    pass

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    pass

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        softplus_x = np.log(1 + np.exp(x))
        tanh_softplus = np.tanh(softplus_x)
        return tanh_softplus + x * (1 - tanh_softplus ** 2) * (1 / (1 + np.exp(-x)))
    pass

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
    pass


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-9, 1.0)  
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true 

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        limit = np.sqrt(6 / (fan_in + fan_out))
        self.Weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.biases = np.zeros((1, fan_out))  

        self.activations = None
        self.dropout_mask = None


    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = h
        z = np.dot(h, self.Weights) + self.biases
        self.activations = self.activation_function.forward(z)

        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.activations.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.activations *= self.dropout_mask

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dropout_rate > 0:
            delta *= self.dropout_mask

        dL_P = delta * self.activation_function.derivative(self.activations)
        dL_dW = np.dot(h.T, dL_P)
        dL_db = np.sum(dL_P, axis=0, keepdims=True)
        delta_prev = np.dot(dL_P, self.Weights.T)
        return dL_dW, dL_db, delta_prev
    
class MultilayerPerceptron:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h = input_data if i == 0 else self.layers[i - 1].activations
            dL_dW, dL_db, delta = layer.backward(h, delta)

            dl_dw_all.insert(0, dL_dW)
            dl_db_all.insert(0, dL_db)

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, rmsprop:bool=False, beta: float=0.9, epsilon:float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
 
        training_losses = []
        validation_losses = []

        cache_Weights = [np.zeros_like(layer.Weights) for layer in self.layers]
        cache_biases = [np.zeros_like(layer.biases) for layer in self.layers]

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                y_pred = self.forward(batch_x)
                batch_loss = np.mean(loss_func.loss(batch_y, y_pred))  # Compute batch loss
                epoch_loss += batch_loss
                num_batches += 1
                
                loss_grad = loss_func.derivative(batch_y, y_pred)
                dl_dw_all, dl_db_all = self.backward(loss_grad, batch_x)

                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        cache_Weights[i] = beta * cache_Weights[i] + (1 - beta) * dl_dw_all[i] ** 2
                        cache_biases[i] = beta * cache_biases[i] + (1 - beta) * dl_db_all[i] ** 2
                        layer.Weights -= learning_rate * dl_dw_all[i] / (np.sqrt(cache_Weights[i]) + epsilon)
                        layer.biases -= learning_rate * dl_db_all[i] / (np.sqrt(cache_biases[i]) + epsilon)
                    else:
                        layer.Weights -= (learning_rate * dl_dw_all[i] / batch_size)
                        layer.biases -= (learning_rate * dl_db_all[i] / batch_size)

            training_losses.append(epoch_loss / num_batches)
            val_pred = self.forward(val_x)
            val_loss = np.mean(loss_func.loss(val_y, val_pred))
            validation_losses.append(val_loss)
            print(f"Epoch {epoch+1}, Training Loss: {training_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")

        return np.array(training_losses), np.array(validation_losses)


