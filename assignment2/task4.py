import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    #learning_rate = .1
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    #neurons_per_layer = [64, 64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    neurons_per_layer = [64, 64, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_2, val_history_2 = trainer.train(num_epochs)
    
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_imp, val_history_imp = trainer.train(num_epochs)

    # Plot 4d,e
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"],
                    "Training Loss - 1 hl", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss - 1 hl")
    utils.plot_loss(train_history_2["loss"],
                    "Training Loss - 2 hl", npoints_to_average=10)
    utils.plot_loss(val_history_2["loss"], "Validation Loss - 2 hl")
    utils.plot_loss(train_history_imp["loss"],
                    "Training Loss - 10 hl", npoints_to_average=10)
    utils.plot_loss(val_history_imp["loss"], "Validation Loss - 10 hl")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.90, 1])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy - 1 hl")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy - 1 hl")
    utils.plot_loss(train_history_2["accuracy"], "Training Accuracy - 10 hl")
    utils.plot_loss(val_history_2["accuracy"], "Validation Accuracy - 10 hl")
    utils.plot_loss(train_history_imp["accuracy"], "Training Accuracy - 10 hl")
    utils.plot_loss(val_history_imp["accuracy"], "Validation Accuracy - 10 hl")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig("task4e.png")
    plt.show()
    
