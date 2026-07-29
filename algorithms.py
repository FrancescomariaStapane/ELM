from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# from ELM_CrossEntropy import *
from codecarbon import OfflineEmissionsTracker
import pandas as pd
import torch
# import hpelm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import f1_score
# import resource
import numpy as np
import gc

INFINITY = 10**12

# training is aborted if it tries to allocate more than this much memory (in bytes) for a single tensor
# adjust empirically
TENSOR_DIMENSION_LIMIT = 0.2 * 1024 ** 3

class MlAlgorithm(ABC):
    def __init__(self, xtr, ytr, xts, yts):
        self.xtr = xtr
        self.ytr = ytr
        self.xts = xts
        self.yts = yts
        self.valid = True
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def refresh(self):
        pass
    @abstractmethod
    def get_default_accuracy(self):
        pass

    def get_final_m_features(self):
        return self.xtr.shape[1]




def new_tracker():
    tracker = OfflineEmissionsTracker(
        country_iso_code="ITA",
        log_level="error",  # Suppresses detailed logging
        save_to_file=False  # Prevents writing to emissions.csv
    )
    return tracker


class CrossEntropyElm(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_neurons, learning_rate, n_features = -1):
        # ytr = torch.from_numpy(ytr.argmax(axis = 1))
        # yts = torch.from_numpy(yts.argmax(axis = 1))
        super().__init__(xtr, ytr, xts, yts)
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.W, self.b = self.random_weights(len(self.xtr[0]), self.n_neurons)
        self.beta = []
        if n_features != -1:
            self.n_original_features = n_features
        else:
            self.n_original_features = len(self.xtr[0])



    def refresh(self):
        gc.collect()
        self.W, self.b = self.random_weights(len(self.xtr[0]), self.n_neurons)

    def learn(self):
        # tracker = new_tracker()
        # tracker.start()
        # time_start = time.time_ns()
        # self.beta = training(self.n_neurons, self.xtr, self.ytr, self.W, self.b, self.learning_rate)
        # tracker.stop()
        # train_time = float((time.time_ns() - time_start) / 10 ** 6)  # in ms
        # training_energy = tracker.final_emissions_data.energy_consumed
        # return train_time, training_energy
        train_time, train_energy, self.beta = run_with_measurement(lambda:
            self.train_cross_entropy())
        return train_time, train_energy

    def train_cross_entropy(self):
        # Calculate H one time because W and b are fixed

        H = sigmoid(self.xtr, self.W, self.b)

        # Inizialize beta as a trainable parameters, the dimensions are (n_neuron, num_classes)
        num_classes = torch.unique(self.ytr).shape[0]

        beta = torch.randn(self.n_neurons, num_classes, requires_grad=True)

        # Define Cross Entropy Loss and Adam
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([beta], lr=self.learning_rate)
        epochs = 500
        for epoch in range(epochs):
            optimizer.zero_grad()  # Reset the gradients

            # Compute the forward pass
            logits = torch.matmul(H, beta)  # The forward pass in ELM is obtained through H * beta

            # Compute the loss
            loss = criterion(logits, self.ytr)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)  # Extract the class with max logit
            accuracy = (predictions == self.ytr).float().mean()

            # Backpropagation and optimazer step
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%')

        return beta  # Return trained parameters
    def test(self):
        # accuracy, f1 = test_model(self.xtr, self.ytr, self.W, self.b, self.beta, silent=False)
        return run_with_measurement(lambda: self.test_model())

    # Evaluate model
    def test_model(self):
        # Compute the forward pass
        H_test = safe_sigmoid(self.xts, self.W, self.b)
        logits = torch.matmul(H_test, self.beta)

        # Extract the class with max logit
        predictions = torch.argmax(logits, dim=1)

        # Compute accuracy
        accuracy = (predictions == self.yts).float().mean()

        # compute f1
        f1 = f1_score(self.yts, predictions, average='weighted')

        # if not silent:
        print(f'Accuracy sui dati di test: {accuracy.item() * 100:.2f}%')
        print(f'F1 score: {f1}')
        return accuracy, f1
    def get_default_accuracy(self):
        most_common_class = torch.mode(self.yts).values
        default_accuracy = (self.yts == most_common_class).sum().item() / self.yts.size(0)
        return default_accuracy

    def get_n_neurons(self):
        return self.n_neurons

    def get_learning_rate(self):
        return self.learning_rate

    def get_n_classes(self):
        # return self.n_classes
        return self.ytr.unique().numel()

    def get_original_n_features(self):
        return self.n_original_features

    # Bias and weights random initialization
    def random_weights(self,input_dimension, n_neuron):
        W = torch.randn(input_dimension, n_neuron)
        b = torch.randn(1, n_neuron)
        return W, b

def mase(y_real, y_pred):
    abs_error = 0
    avg_real = 0
    for i in range(y_real.shape[0]):
        abs_error += np.abs(y_real[i] - y_pred[i])
        avg_real += y_real[i]
    mae = abs_error / y_real.shape[0]
    avg_real = avg_real / y_real.shape[0]
    mad = 0
    for i in range(y_real.shape[0]):
        mad += np.abs(y_real[i] - avg_real)
    mad = mad / y_real.shape[0]
    return mae / mad


def sse(y_real, y_predict):
    sse = 0
    for i in range(y_real.shape[0]):
        sse += (y_real[i] - y_predict[i])**2
    return sse

def safe_sigmoid(X, W, b):
    z = safe_matmul(X, W) + b
    return torch.sigmoid(z)

def sigmoid(X, W, b):
    z = torch.matmul(X, W) + b
    return torch.sigmoid(z)


class RegularizedElm(MlAlgorithm):
    def __init__(self, xtr, ytr, xts, yts, n_neurons, lmbda, n_features=-1):
        super().__init__(xtr, ytr, xts, yts)
        try:
            # target normalization
            self.interval = torch.cat([ytr, yts]).max() - torch.cat([ytr, yts]).min()
            self.target_normalization = True
            self.lmbda = lmbda
            if self.target_normalization:
                self.y_mean = self.ytr.mean()
                self.y_std = self.ytr.std()
                self.ytr = (self.ytr - self.y_mean) / self.y_std
            self.input_dimension = len(self.xtr[0])
            self.n_neurons = n_neurons
            self.W = torch.randn(self.input_dimension, self.n_neurons)
            self.b = torch.randn(1, self.n_neurons)
            self.h = safe_sigmoid(self.xtr, self.W, self.b)
            if n_features != -1:
                self.n_original_features = n_features
            else:
                self.n_original_features = len(self.xtr[0])
        except MemoryError:
            self.valid = False

    def learn(self):
        if self.valid:
            try:
                h_transpose = self.h.T
                train_time, train_energy, self.beta = run_with_measurement(lambda:
                    safe_matmul(
                    torch.inverse(safe_matmul(h_transpose, self.h) + self.lmbda * torch.eye(self.n_neurons)),
                    safe_matmul(h_transpose, self.ytr)
                ))
                A = safe_matmul(h_transpose, self.h) + self.lmbda * torch.eye(self.n_neurons)
                b = safe_matmul(h_transpose, self.ytr)
                self.beta = torch.linalg.solve(A, b)
                return train_time, train_energy
            except MemoryError:
                self.valid = False
        return INFINITY, INFINITY

    def test(self):
        if self.valid:
            h = safe_sigmoid(self.xts, self.W, self.b)
            test_time, test_energy, y_pred = run_with_measurement(lambda: safe_matmul(h,self.beta))
            if self.target_normalization:
                y_pred = y_pred * self.y_std + self.y_mean
            loss = sse(self.yts , y_pred)
            mean_loss = loss / self.yts.size(0)
            rae_ = mase(self.yts, y_pred)
            return test_time, test_energy, (mean_loss, rae_)
        return INFINITY, INFINITY, (INFINITY, INFINITY)
    def refresh(self):
            gc.collect()
            self.valid = True
            self.W = torch.randn(self.input_dimension, self.n_neurons)
            self.b = torch.randn(1, self.n_neurons)
            try:
                self.h = safe_sigmoid(self.xtr, self.W, self.b)
            except MemoryError:
                self.valid = False

    def get_default_accuracy(self):
        pass

    def get_n_neurons(self):
        return self.n_neurons

    def get_lambda(self):
        return self.lmbda

    def get_original_n_features(self):
        return self.n_original_features


def numpy_to_float_tensor(items):
    items = torch.from_numpy(items)
    items = items.type('torch.FloatTensor')
    return items
def run_with_measurement(code):
    tracker = new_tracker()
    tracker.start()
    time_start = time.time_ns()
    results = code()
    tracker.stop()
    time_passed = float((time.time_ns() - time_start) / 10 ** 6)  # in ms
    energy_used = tracker.final_emissions_data.energy_consumed
    return time_passed, energy_used, results

def estimate_matmul_memory(A: torch.Tensor, B: torch.Tensor, safety_factor=1.3):
    dtype_size = torch.tensor([], dtype=A.dtype).element_size()
    size_A = A.numel() * dtype_size
    size_B = B.numel() * dtype_size
    m, n, n2, p= 1,1,1,1
    if(A.dim() == 1):
        m = A.shape[0]
    else:
        m, n = A.shape[0], A.shape[1]
    if (B.dim() == 1):
        p = B.shape[0]
    else:
        n2, p = B.shape[0], B.shape[1]
    # assert n == n2, "incompatible shapes"
    size_out = m * p * dtype_size
    total = (size_A + size_B + size_out) * safety_factor
    return total

def relu(X, W, b):
    return torch.relu(safe_matmul(X, W) + b)

def safe_matmul(A: torch.Tensor, B:torch.Tensor, safety_factor=1.3, limit=TENSOR_DIMENSION_LIMIT):
    if(estimate_matmul_memory(A,B, safety_factor) > limit):
        print("aborting matmul, ", (estimate_matmul_memory(A,B, safety_factor)))
        raise MemoryError
    return torch.matmul(A, B)