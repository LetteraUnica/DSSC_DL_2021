import pylab as pl


def plot_train_test(train_losses: list, test_losses: list):
    """
    Plots the training and test loss in two separate graphs
    """
    import seaborn as sns
    sns.set_theme()
    epochs = list(range(1, len(train_losses)+1))
    fig, axes = pl.subplots(1, 2, figsize=(12,4))
    axes[0].plot(epochs, train_losses, label="train loss")
    axes[1].plot(epochs, test_losses, label="test misclassification rate")
    axes[0].title.set_text("Training loss")
    axes[1].title.set_text("Test misclassification rate")
    axes[0].set_xlabel("Number of epochs")
    axes[1].set_xlabel("Number of epochs")