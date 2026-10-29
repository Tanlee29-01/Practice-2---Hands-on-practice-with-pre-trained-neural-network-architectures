"""Training and validation loop utilities."""

import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader,
        learning_rate=0.001,
        optimizer_name="Adam",
        device=None,
        weight_decay=0.0,
    ):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = device
        self.model = model.to(self.device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Hàm mất mát cho bài toán phân loại.
        self.criterion = nn.CrossEntropyLoss()

        # Truyền weight_decay vào optimizer để thử nghiệm L2.
        if optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError("Optimizer_name must be 'SGD' or 'Adam'")

    def training_loop(self):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Xóa gradient của bước trước.
            self.optimizer.zero_grad()

            # Dự đoán và tính loss.
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Tính gradient và cập nhật trọng số.
            loss.backward()
            self.optimizer.step()

            # Cộng dồn loss và số dự đoán đúng.
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def fit(self, num_epochs):
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_losses": [],
            "test_accuracy": [],
        }

        for epoch in range(num_epochs):
            train_loss, train_acc = self.training_loop()
            test_loss, test_acc = self.evaluate()

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["test_losses"].append(test_loss)
            history["test_accuracy"].append(test_acc)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Losses: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}%"
            )

        return history