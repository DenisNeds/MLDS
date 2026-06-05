import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ANNClassificationTorch(nn.Module):
    
    def __init__(self, n_features, n_classes, layers = None, seed = None):
        super().__init__()
        if layers is None:
            layers = [16]
        if seed is not None:
            torch.manual_seed(seed)    

        self.network = nn.Sequential()
        layers_size = [n_features] + layers + [n_classes]

        for i in range(len(layers_size) - 1):
            linear = nn.Linear(layers_size[i], layers_size[i + 1])

            limit = np.sqrt(6.0 / (layers_size[i] + layers_size[i + 1]))
            nn.init.uniform_(linear.weight, -limit, limit)
            nn.init.zeros_(linear.bias)

            self.network.add_module(f"fc_{i}", linear)

            if i < len(layers_size) - 2:
                self.network.add_module(f"f_sigmoid_{i}", nn.Sigmoid())

    def forward(self,x):
        return self.network(x)


class PytorchTrainer:
    def __init__(self, model, learning_rate = 0.1, epochs = 1000, batch_size = 32, momentum = 0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum

        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate, momentum=momentum)
        self.loss = nn.CrossEntropyLoss()

    def fit(self, x, y):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype = torch.long)  

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy()
                                          
