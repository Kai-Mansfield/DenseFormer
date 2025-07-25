import torch
import torch.nn as nn
import torch.optim as optim

# Check for at least 2 GPUs
assert torch.cuda.device_count() >= 2, "You need at least 2 GPUs"

# Toy dataset: simple regression (y = 2x + 1)
x = torch.linspace(-1, 1, 100).view(-1, 1)
y = 2 * x + 1 + 0.1 * torch.randn_like(x)

# Move data to cuda:0 where the input layer will be
x = x.to('cuda:0')
y = y.to('cuda:0')

# Define a simple 4-layer model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # First half on cuda:0
        self.fc1 = nn.Linear(1, 16).to('cuda:0')
        self.relu1 = nn.ReLU().to('cuda:0')
        self.fc2 = nn.Linear(16, 32).to('cuda:0')

        # Second half on cuda:1
        self.relu2 = nn.ReLU().to('cuda:1')
        self.fc3 = nn.Linear(32, 16).to('cuda:1')
        self.fc4 = nn.Linear(16, 1).to('cuda:1')

    def forward(self, x):
        # Stage 1 on cuda:0
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        # Move output to cuda:1
        x = x.to('cuda:1')

        # Stage 2 on cuda:1
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

model = SimpleModel()

# Define loss and optimizer (parameters from both GPUs)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()

    # Forward pass
    output = model(x)

    # Compute loss (also move labels to cuda:1)
    loss = criterion(output, y.to('cuda:1'))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
