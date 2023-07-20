import torch

class SimpleCNN(torch.nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_hidden_layers: int,
            use_batchnormalization: bool,
            num_classes: int,
            kernel_size: int = 3,
            activation_function: torch.nn.Module = torch.nn.ReLU()
    ):
        super().__init__()
        hidden_layers = torch.nn.ModuleList()
        first_channel = input_channels
        for _ in range(num_hidden_layers):
            # Add a CNN layer
            layer = torch.nn.Conv2d(in_channels=first_channel, out_channels=hidden_channels, kernel_size=kernel_size, padding = kernel_size//2)
            hidden_layers.append(layer)
            # Add activation module to list of modules
            if use_batchnormalization is True:
                hidden_layers.append(torch.nn.BatchNorm2d(hidden_channels))
            hidden_layers.append(activation_function)

            first_channel = hidden_channels

        self.hidden_layers = hidden_layers
        self.flatten = torch.nn.Flatten()
        self.output_layer = torch.nn.Linear(in_features=hidden_channels*64*64, out_features=num_classes)

    def forward(self, input_images: torch.Tensor):
        for layer in self.hidden_layers:
            input_images = layer(input_images)

        input_images = self.flatten(input_images)
        # hidden_features = hidden_features.view(hidden_features.size(0), -1)
        # Apply last layer (=output layer)
        output = self.output_layer(input_images)
        return output



if    __name__    ==    "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=torch.nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(output)



