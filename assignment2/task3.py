import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    #learning_rate = .1
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
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

    ### Task 3. ###
    # Add improved weight
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_w, val_history_w = trainer.train(num_epochs)

    # Add improved sigmoid
    use_improved_sigmoid = True
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_s, val_history_s = trainer.train(num_epochs)

    # Add momentum
    use_momentum = True
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_m, val_history_m = trainer.train(num_epochs)


    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_w["loss"], "Weight init improved", npoints_to_average=10)
    utils.plot_loss(
        train_history_s["loss"], "Weight+sigmoid improved", npoints_to_average=10)
    utils.plot_loss(
        train_history_m["loss"], "Weight+sigmoid+momentum", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Training Cross Entropy Loss")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .98])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_w["accuracy"], "Weight init improved")
    utils.plot_loss(
        val_history_s["accuracy"], "Weight+sigmoid improved")
    utils.plot_loss(
        val_history_m["accuracy"], "Weight+sigmoid+momentum")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #plt.savefig("task3.png")
    plt.show()
