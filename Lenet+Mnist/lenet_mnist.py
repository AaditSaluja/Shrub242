import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Input: 1x28x28, Output: 6x24x24
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 6x12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output: 16x8x8
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten to 16x4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        self.layer_outputs = {}  # Dictionary to store outputs at each layer
        x = torch.relu(self.conv1(x))
        self.layer_outputs['conv1'] = x
        x = self.pool(x)
        self.layer_outputs['pool1'] = x
        x = torch.relu(self.conv2(x))
        self.layer_outputs['conv2'] = x
        x = self.pool(x)
        self.layer_outputs['pool2'] = x
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        self.layer_outputs['fc1'] = x
        x = torch.relu(self.fc2(x))
        self.layer_outputs['fc2'] = x
        x = self.fc3(x)
        self.layer_outputs['fc3'] = x
        return x

# Calculate accuracy at every layer
def layer_wise_accuracy(model, device, test_loader):
    model.eval()
    correct = {layer: 0 for layer in model.layer_outputs.keys()}
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            
            for layer_name, layer_output in model.layer_outputs.items():
                if len(layer_output.shape) == 4:  # Convolutional layers
                    # Global average pooling to reduce spatial dimensions
                    layer_output = layer_output.mean(dim=[2, 3])
                
                if layer_output.shape[1] > 10:  # Fully connected or intermediate layer
                    layer_output = layer_output[:, :10]  # Only keep 10 classes for comparison
                
                preds = torch.argmax(layer_output, dim=1)
                correct[layer_name] += (preds == labels).sum().item()
            total += labels.size(0)
    
    for layer_name, num_correct in correct.items():
        print(f"Accuracy at {layer_name}: {100 * num_correct / total:.2f}%")

# Train the model
def train_model(model, device, train_loader, optimizer, criterion, epochs=25):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
train_model(model, device, train_loader, optimizer, criterion, epochs=25)

print("Testing the model...")
layer_wise_accuracy(model, device, test_loader)
