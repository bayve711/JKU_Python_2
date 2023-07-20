import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):
    
    def __init__(
            self,
            input_neurons: int,
            hidden_neurons: int,
            output_neurons: int,
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.activation_function = activation_function
        
        self.input_layer = nn.Linear(self.input_neurons, self.hidden_neurons)
        self.hidden_layer_1 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.hidden_layer_2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.output_layer = nn.Linear(self.hidden_neurons, self.output_neurons)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    input = torch.randn(1, 10)
    output = simple_network(input)
    print(output)
