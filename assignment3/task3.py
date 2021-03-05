import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots, ExampleModel


class Net1(nn.Module):
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

        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*4*4#32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_filters*2),
            nn.BatchNorm1d(num_filters*2),
            nn.ReLU(),
            nn.Linear(num_filters*2, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    
class Net2(nn.Module):
    # Droput added
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

        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.05),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.05)
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*4*4#32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_filters*2),
            nn.BatchNorm1d(num_filters*2),
            nn.LeakyReLU(),
            nn.Linear(num_filters*2, num_classes),
            nn.BatchNorm1d(num_classes)
        )
        #self.feature_extractor.apply(self.init_weights)
        #self.classifier.apply(self.init_weights)

    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    def init_weights(self, m):
        if type(m) == (nn.Linear or nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    # Network 1: (worst one)
    """print("Network 1:")
    model1 = Net1(image_channels=3, num_classes=10)
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model1,
        dataloaders
    )
    trainer1.train()
    create_plots(trainer1, "task3_net2")
    _, train_acc = compute_loss_and_accuracy(
        trainer1.dataloader_train, trainer1.model, trainer1.loss_criterion
    )
    _, val_acc = compute_loss_and_accuracy(
        trainer1.dataloader_val, trainer1.model, trainer1.loss_criterion
    )
    _, test_acc = compute_loss_and_accuracy(
        trainer1.dataloader_test, trainer1.model, trainer1.loss_criterion
    )
    print("Final train accuracy = ", train_acc.item())
    print("Final validation accuracy = ", val_acc.item())
    print("Final test accuracy = ", test_acc.item())"""

    # Network 2:
    print("Network 2:")
    model2 = Net2(image_channels=3, num_classes=10)
    trainer2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model2,
        dataloaders
    )
    trainer2.train()
    trainer2.load_best_model()
    #create_plots(trainer2, "task3_net2")
    create_plots(trainer2, "task3_net2_improved")
    
    _, train_acc = compute_loss_and_accuracy(
        trainer2.dataloader_train, trainer2.model, trainer2.loss_criterion
    )
    """_, val_acc = compute_loss_and_accuracy(
        trainer2.dataloader_val, trainer2.model, trainer2.loss_criterion
    )
    _, test_acc = compute_loss_and_accuracy(
        trainer2.dataloader_test, trainer2.model, trainer2.loss_criterion
    )"""
    print("Final train accuracy = ", train_acc.item())
    print("Final validation accuracy = ", trainer2.validation_history["accuracy"].popitem()[1].item())
    print("Final test accuracy = ", trainer2.test_history["accuracy"].popitem()[1].item())
