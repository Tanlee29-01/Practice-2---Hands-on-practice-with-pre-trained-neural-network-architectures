from src.data_module import CIFAR10DataModule


data_module = CIFAR10DataModule(
    data_dir="./data",
    batch_size=64,
    image_size=224,
)

data_module.setup()

train_loader = data_module.train_dataloader()
test_loader = data_module.test_dataloader()