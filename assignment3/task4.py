import pathlib
import matplotlib.pyplot as plt
import utils
import torchvision
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots, ExampleModel


class ResNet(nn.Module):
    # added batch normalization
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fully_connected = nn.Linear(224, 10)
        

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        #batch_size = x.shape[0]
        out = self.model(x)
        return out



if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ResNet(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    trainer.load_best_model()
    create_plots(trainer, "task4")
    print("Final validation accuracy = ", trainer.validation_history["accuracy"].popitem()[1].item())
    print("Final test accuracy = ", trainer.test_history["accuracy"].popitem()[1].item())
