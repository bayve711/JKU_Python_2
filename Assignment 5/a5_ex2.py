import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from a4_ex1 import SimpleNetwork
from dataset import get_dataset
import matplotlib.pyplot as plt

def training_loop(network: torch.nn.Module, train_data: torch.utils.data.Dataset, eval_data: torch.utils.data.Dataset, num_epochs: int, show_progress: bool = False ) -> tuple[list, list]:
    #optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)

    #DataLoaders
    training_loader = DataLoader(
        train_data,
        shuffle=False,
        batch_size=32,
        num_workers=0
    )
    eval_loader = DataLoader(
        eval_data,
        shuffle=False,
        batch_size=32,
        num_workers=0
    )
    train_losses = []
    eval_losses = []
    loss_function = nn.MSELoss()
    min_loss = float("inf")
    no_improvement_streak = 0

    for _ in range(num_epochs):
        if show_progress:
            training_loader = tqdm(training_loader)
        all_losses_1 = 0
        network.train()
        for input, target in training_loader:
            output = network(input).squeeze()
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_losses_1 += loss.item()
        loss_avg_1 = all_losses_1 / len(training_loader)
        train_losses.append(loss_avg_1)
        all_losses_2 = 0
        network.eval()
        for input, target in eval_loader:
            output = network(input).squeeze()
            loss = loss_function(output, target)
            all_losses_2 += loss.item()
        loss_avg_2 = all_losses_2 / len(eval_loader)
        eval_losses.append(loss_avg_2)
        if loss_avg_2 < min_loss:
            min_loss = loss_avg_2
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
            if no_improvement_streak == 3:
                break

    return train_losses, eval_losses

def plot_losses(train_losses, eval_losses):
    x_axis = range(0, len(train_losses))
    plt.plot(x_axis, train_losses, label='Training curve')
    plt.plot(x_axis, eval_losses, label='Evaluation curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE-Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100, show_progress=True)
    plot_losses(train_losses, eval_losses)


