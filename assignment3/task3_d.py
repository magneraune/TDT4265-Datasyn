import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots, ExampleModel




def create_plots_2(trainer1: Trainer, trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer1.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer1.validation_history["loss"], label="Validation loss")
    utils.plot_loss(trainer2.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer1.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, data_aug=False)
    # Without data augmentation:
    model1 = ExampleModel(image_channels=3, num_classes=10)
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model1,
        dataloaders
    )
    trainer1.train()
    # With data augmentation:
    dataloaders = load_cifar10(batch_size)
    model2 = ExampleModel(image_channels=3, num_classes=10)
    trainer2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model2,
        dataloaders
    )
    trainer2.train()
    create_plots_2(trainer1, trainer2, "task3d_plots")
    
