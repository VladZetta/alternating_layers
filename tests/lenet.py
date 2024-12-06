import time

import torch
import torch.nn as nn


class Runner:
    def __init__(self, model, loss_fn, metrics_fn, random_seed=0, device="cuda:6", verbose=0):
        self.random_seed = random_seed
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn # nn.BCELoss() for FCNN loss should be a class argument
        self.metrics_fn = metrics_fn
        self.verbose = verbose
        self.set_seed()

    def set_seed(self):
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

    def test(self, test_loader):
        self.model.eval()
        running_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                running_loss += self.loss_fn(outputs, y).item() * y.size(0)
                correct += self.metrics_fn(outputs, y)
                total += y.size(0)
        return running_loss / total, correct / total

    
    def train_one_epoch(self, train_loader, test_loader, optimizer1, optimizer2=None):
        running_loss = 0
        total = 0
        correct = 0
        train_start_time = time.time()

        self.model.train()
        
        for idx, (X, y) in enumerate(train_loader):
                
            X = X.to(self.device)
            y = y.to(self.device)

            def closure():
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                loss.backward(retain_graph=True)
                return loss

            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            running_loss += loss.item() * y.size(0) 
            correct += self.metrics_fn(outputs, y)
            total += y.size(0)

            if (self.verbose >= 3) and ((idx + 1) % 50 == 0): 
                test_start_time = time.time()
                loss_test, accuracy_test = self.test(test_loader)
                print(f"\tStep: {idx+1}/{len(train_loader)} Train loss: {loss.item():.4f} Test loss: {loss_test:.4f} Train accuracy: {self.metrics_fn(outputs, y) / y.size(0):.4f} Test accuracy: {accuracy_test:.4f}")
                test_time = time.time() - test_start_time # time spent on testing
                train_start_time += test_time  # subtract testing time from training time



            if optimizer2 is not None:
                optimizer2.zero_grad()
                optimizer2.step(closure)

            optimizer1.zero_grad()
            optimizer1.step(closure)

        epoch_time = time.time() - train_start_time
        return running_loss / total, correct / total, epoch_time

        
    def train(self, num_epochs, train_loader, test_loader, optimizer1, optimizer2=None):
        loss_train_vals = []
        loss_test_vals = []
        accuracy_train_vals = []
        accuracy_test_vals = []
        epoch_times = []

        loss_test, accuracy_test = self.test(test_loader)
        loss_test_vals.append(loss_test)
        accuracy_test_vals.append(accuracy_test)

        loss_train, accuracy_train = self.test(train_loader)
        loss_train_vals.append(loss_train)
        accuracy_train_vals.append(accuracy_train)

        epoch_times.append(0.)

        if self.verbose >= 1:
            print(f"Epoch 0/{num_epochs} Train loss: {loss_train:.4f} Test loss: {loss_test:.4f} Train accuracy: {accuracy_train:.4f} Test accuracy: {accuracy_test:.4f}")
        
        for epoch in range(num_epochs):
            loss_train, accuracy_train, epoch_time = self.train_one_epoch(train_loader=train_loader, test_loader=test_loader, optimizer1=optimizer1, optimizer2=optimizer2)

            loss_train_vals.append(loss_train)
            accuracy_train_vals.append(accuracy_train)
            loss_test, accuracy_test = self.test(test_loader)
            loss_test_vals.append(loss_test)
            accuracy_test_vals.append(accuracy_test)
            epoch_times.append(epoch_time)

            if self.verbose >= 2:
                print(f"Epoch: {epoch+1}/{num_epochs} Train loss: {loss_train:.4f} Test loss: {loss_test:.4f} Train accuracy: {accuracy_train:.4f} Test accuracy: {accuracy_test:.4f}")
            elif self.verbose >= 1 and (epoch + 1 == num_epochs):
                print(f"Epoch: {epoch+1}/{num_epochs} Train loss: {loss_train:.4f} Test loss: {loss_test:.4f} Train accuracy: {accuracy_train:.4f} Test accuracy: {accuracy_test:.4f}")
        results = {
            "loss": {
                "train": loss_train_vals,
                "test": loss_test_vals,
            },
            "accuracy": {
                "train": accuracy_train_vals,
                "test": accuracy_test_vals,
            },
            "time": epoch_times,
        }
        return results
            