"""TensorBoard logging utilities."""
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

# Thư mục gốc của project (cha của thư mục src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TensorBoardLogger:
    """Ghi log TensorBoard và quản lý checkpoint (trọng số mô hình).

    Ví dụ:
        logger = TensorBoardLogger(experiment_name="resnet18_adam")
        logger.add_graph(model, sample_images)
        logger.add_image("Samples", grid_tensor)
        for epoch in range(num_epochs):
            logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)
            logger.save_best_model(model, test_acc, optimizer, epoch)
        logger.close()
    """

    def __init__(self, log_dir=None, experiment_name=None, checkpoint_dir=None):
        """
        Args:
            log_dir: Thư mục gốc lưu log TensorBoard (mặc định: <project>/runs).
            experiment_name: Tên thí nghiệm (tạo thư mục con). Nếu None sẽ dùng timestamp.
            checkpoint_dir: Thư mục lưu file .pth (mặc định: <project>/checkpoints).
        """
        log_dir = Path(log_dir) if log_dir is not None else PROJECT_ROOT / "runs"
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.experiment_name = experiment_name
        self.log_dir = log_dir / experiment_name
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else PROJECT_ROOT / "checkpoints"
        )

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_metric = float("-inf")

    # ------------------------------------------------------------------ #
    # TensorBoard logging
    # ------------------------------------------------------------------ #
    def add_scalar(self, tag, value, step):
        """Ghi một giá trị số (Loss, Accuracy, learning rate, ...)."""
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag, tag_scalar_dict, step):
        """Ghi nhiều giá trị trên cùng một biểu đồ (ví dụ Train vs Test Loss)."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def add_graph(self, model, input_to_model):
        """Ghi cấu trúc (computational graph) của mô hình.

        Args:
            model: nn.Module cần vẽ graph.
            input_to_model: Tensor mẫu, ví dụ shape (N, 3, 224, 224).
        """
        was_training = model.training
        model.eval()
        try:
            device = next(model.parameters()).device
            with torch.no_grad():
                self.writer.add_graph(model, input_to_model.to(device))
        except Exception as exc:  # add_graph có thể lỗi với một số kiến trúc
            warnings.warn(f"Không thể ghi model graph lên TensorBoard: {exc}")
        finally:
            model.train(was_training)

    def add_image(self, tag, img_tensor, step=0):
        """Ghi một ảnh (hoặc một grid ảnh đã ghép) lên TensorBoard."""
        self.writer.add_image(tag, img_tensor, step)

    def add_image_grid(
        self,
        tag,
        img_batch,
        nrow=8,
        step=0,
        unnormalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        """Hien thi luoi hinh anh mau trong tap du lieu len TensorBoard (Chuong 6.1).

        Su dung torchvision.utils.make_grid va tu dong khu chuan hoa (un-normalize)
        de tai tao mau sac anh RGB goc trung thuc nhat.
        """
        import torchvision.utils as vutils

        tensor = img_batch.detach().clone().cpu()
        if unnormalize:
            for c in range(3):
                tensor[:, c, :, :] = tensor[:, c, :, :] * std[c] + mean[c]
            tensor = torch.clamp(tensor, 0.0, 1.0)

        grid = vutils.make_grid(tensor, nrow=nrow)
        self.writer.add_image(tag, grid, step)


    def add_images(self, tag, img_batch, step=0):
        """Ghi cả batch ảnh, shape (N, C, H, W)."""
        self.writer.add_images(tag, img_batch, step)

    def add_histogram(self, tag, values, step):
        """Ghi phân phối giá trị (ví dụ trọng số, gradient)."""
        self.writer.add_histogram(tag, values, step)

    def add_model_histograms(self, model, step):
        """Ghi histogram của toàn bộ tham số mô hình."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param.detach().cpu(), step)
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), step)

    def add_text(self, tag, text, step=0):
        """Ghi văn bản (ví dụ cấu hình siêu tham số)."""
        self.writer.add_text(tag, text, step)

    def log_epoch(self, epoch, train_loss, train_acc, test_loss=None, test_acc=None):
        """Ghi toàn bộ chỉ số của một epoch (tiện dùng trong vòng lặp fit)."""
        self.add_scalar("Loss/train", train_loss, epoch)
        self.add_scalar("Accuracy/train", train_acc, epoch)
        if test_loss is not None:
            self.add_scalar("Loss/test", test_loss, epoch)
            self.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        if test_acc is not None:
            self.add_scalar("Accuracy/test", test_acc, epoch)
            self.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)

    def log_history(self, history):
        """Ghi lại toàn bộ `history` dict do `ModelTrainer.fit()` trả về."""
        n_epochs = len(history.get("train_loss", []))
        for epoch in range(n_epochs):
            self.log_epoch(
                epoch + 1,
                history["train_loss"][epoch],
                history["train_accuracy"][epoch],
                history["test_losses"][epoch] if "test_losses" in history else None,
                history["test_accuracy"][epoch] if "test_accuracy" in history else None,
            )

    def flush(self):
        self.writer.flush()

    def close(self):
        """Đóng writer, đảm bảo log được ghi hết xuống đĩa."""
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------------------ #
    # Checkpointing (quản lý trọng số)
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, model, optimizer=None, epoch=None, metrics=None, filename=None):
        """Lưu trọng số mô hình (kèm optimizer, epoch, metrics) ra file .pth.

        Returns:
            Đường dẫn file đã lưu.
        """
        if filename is None:
            suffix = f"_epoch{epoch}" if epoch is not None else ""
            filename = f"{self.experiment_name}{suffix}.pth"
        path = self.checkpoint_dir / filename

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(state, path)
        return path

    def save_best_model(self, model, metric, optimizer=None, epoch=None, higher_is_better=True):
        """Chỉ lưu checkpoint khi `metric` tốt hơn giá trị tốt nhất trước đó.

        Args:
            metric: Giá trị theo dõi (ví dụ test accuracy hoặc test loss).
            higher_is_better: True nếu metric càng lớn càng tốt (Accuracy),
                False nếu càng nhỏ càng tốt (Loss).

        Returns:
            True nếu đã lưu model mới tốt nhất, ngược lại False.
        """
        score = metric if higher_is_better else -metric
        if score > self.best_metric:
            self.best_metric = score
            path = self.save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics={"best_metric": metric},
                filename=f"{self.experiment_name}_best.pth",
            )
            print(f"[Checkpoint] Saved best model (metric={metric:.4f}) -> {path}")
            return True
        return False

    def load_checkpoint(self, model, path, optimizer=None, device=None):
        """Nạp trọng số từ file checkpoint vào model (và optimizer nếu có).

        Returns:
            dict checkpoint (chứa epoch, metrics, ...).
        """
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            path = self.checkpoint_dir / path
        if device is None:
            device = next(model.parameters()).device

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
