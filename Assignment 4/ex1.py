import torch
class SimpleNetwork(torch.nn.Module):
    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int,
                 activation_function: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()
        self.input_layer = torch.nn.Linear(in_features=input_neurons, out_features=hidden_neurons)
        self.hidden_layer1 = torch.nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.hidden_layer2 = torch.nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.output_layer = torch.nn.Linear(in_features=hidden_neurons, out_features=output_neurons)
        self.function = activation_function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.function(x)
        x = self.hidden_layer1(x)
        x = self.function(x)
        x = self.hidden_layer2(x)
        x = self.function(x)
        result = self.output_layer(x)
        return result

if    __name__    ==    "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    input = torch.randn(1, 10)
    output = simple_network(input)
    print(output)







