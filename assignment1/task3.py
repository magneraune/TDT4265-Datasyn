import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    logits = model.forward(X)
    outputs = np.zeros_like(logits)
    outputs[np.arange(len(logits)), logits.argmax(1)] = 1
    accuracy = np.mean((outputs==targets).all(1))
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        logits = self.model.forward(X_batch)
        self.model.backward(X_batch, logits, Y_batch)
        self.model.w -= self.learning_rate*self.model.grad
        loss = cross_entropy_loss(Y_batch, logits)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.
    
    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    
    # Train a model with L2 regularization (task 4b)
    l2_reg_lambda = 1
    model1 = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer1 = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer1.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    # images together
    """weight = np.zeros((28*2,28*10))
    #weight1 = np.zeros((28,28*10))
    for i in range(10):
        weight[:28,i*28:(i+1)*28] = np.reshape(trainer.model.w[1:,i], (28, 28))
        weight[28:,i*28:(i+1)*28] = np.reshape(trainer1.model.w[1:,i], (28, 28))
    plt.imshow(weight, cmap='gray')
    plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")
    plt.show()"""
    # images seperate
    weight = np.zeros((28,28*10))
    weight1 = np.zeros((28,28*10))
    for i in range(10):
        weight[:,i*28:(i+1)*28] = np.reshape(trainer.model.w[1:,i], (28, 28))
        weight1[:,i*28:(i+1)*28] = np.reshape(trainer1.model.w[1:,i], (28, 28))
    plt.imshow(weight, cmap='gray')
    plt.imsave("task4b_softmax_weight0.png", weight, cmap="gray")
    plt.show()
    plt.imshow(weight1, cmap='gray')
    plt.imsave("task4b_softmax_weight1.png", weight, cmap="gray")
    plt.show()
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    weight_norm = np.zeros((4,1))
    plt.ylim([0.70, .93])
    i=0
    for l2_lambda in l2_lambdas:
        model = SoftmaxModel(l2_lambda)
        # Train model
        trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        #utils.plot_loss(train_history["accuracy"], "Training Accuracy")
        utils.plot_loss(val_history["accuracy"], ("Validation Accuracy: lamda = ", str(l2_lambda)) )
        weight_norm[i] = np.sum(model.w*model.w)
        i+=1
        print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_lambdas, np.sqrt(weight_norm))
    plt.ylabel("weight_norm")
    plt.xlabel("Lamdas")
    #plt.savefig("task4d_l2_reg_norms.png")
    plt.show()
    