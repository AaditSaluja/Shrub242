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

# Define the LeNet model
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
        self.layer_outputs = {}

        # First convolution
        x = torch.relu(self.conv1(x))
        self.layer_outputs['conv1'] = x

        # First pooling
        x = self.pool(x)
        self.layer_outputs['pool1'] = x

        # Second convolution
        x = torch.relu(self.conv2(x))
        self.layer_outputs['conv2'] = x

        # Second pooling
        x = self.pool(x)
        self.layer_outputs['pool2'] = x

        # Fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        self.layer_outputs['fc1'] = x

        x = torch.relu(self.fc2(x))
        self.layer_outputs['fc2'] = x

        x = self.fc3(x)
        return x

# Define an auxiliary model for each convolutional layer
class AuxiliaryModel(nn.Module):
    def __init__(self, input_dim):
        super(AuxiliaryModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# Training function for the main model (LeNet)
def train_lenet(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Only final output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Main Model Loss: {running_loss / len(train_loader):.4f}")

# Training function for auxiliary models
def train_auxiliary_models(model, auxiliary_models, device, train_loader, optimizers, criteria, epochs=10):
    for param in model.parameters():
        param.requires_grad = False  # Freeze LeNet weights

    model.eval()  # LeNet remains in evaluation mode
    for epoch in range(epochs):
        running_losses = {layer: 0.0 for layer in auxiliary_models.keys()}
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            _ = model(images)  # Forward pass to populate layer_outputs

            # Train auxiliary models
            for layer_name, aux_model in auxiliary_models.items():
                aux_optimizer = optimizers[layer_name]
                aux_criterion = criteria[layer_name]

                aux_output = aux_model(model.layer_outputs[layer_name].detach())
                aux_loss = aux_criterion(aux_output, labels)

                aux_optimizer.zero_grad()
                aux_loss.backward()
                aux_optimizer.step()

                running_losses[layer_name] += aux_loss.item()

        print(f"Epoch {epoch + 1}, " +
              ", ".join([f"{layer} Loss: {running_losses[layer] / len(train_loader):.4f}" for layer in auxiliary_models.keys()]))

# Evaluation function for both the main model and auxiliary models
# def evaluate_models(model, auxiliary_models, device, test_loader):
#     model.eval()
#     for aux_model in auxiliary_models.values():
#         aux_model.eval()

#     correct = {'main': 0}
#     correct.update({layer: 0 for layer in auxiliary_models.keys()})
#     total = 0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)  # Final output

#             correct['main'] += (torch.argmax(outputs, dim=1) == labels).sum().item()

#             for layer_name, aux_model in auxiliary_models.items():
#                 aux_output = aux_model(model.layer_outputs[layer_name])
#                 correct[layer_name] += (torch.argmax(aux_output, dim=1) == labels).sum().item()

#             total += labels.size(0)

#     for key, num_correct in correct.items():
#         print(f"Accuracy for {key}: {100 * num_correct / total:.2f}%")

def evaluate_models(model, auxiliary_models, device, test_loader):
    model.eval()
    for aux_model in auxiliary_models.values():
        aux_model.eval()

    num_classes = 10  # MNIST has 10 classes (digits 0-9)
    class_correct = {'main': [0 for _ in range(num_classes)]}
    class_total = {'main': [0 for _ in range(num_classes)]}
    class_correct.update({layer: [0 for _ in range(num_classes)] for layer in auxiliary_models.keys()})
    class_total.update({layer: [0 for _ in range(num_classes)] for layer in auxiliary_models.keys()})

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Final output

            # Main model predictions
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct['main'][label] += c[i].item()
                class_total['main'][label] += 1

            # Auxiliary models predictions
            for layer_name, aux_model in auxiliary_models.items():
                aux_output = aux_model(model.layer_outputs[layer_name])
                _, predicted = torch.max(aux_output, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[layer_name][label] += c[i].item()
                    class_total[layer_name][label] += 1

    # Printing class-wise accuracy
    for key in class_correct.keys():
        print(f"\nClass-wise accuracy for {key}:")
        for i in range(num_classes):
            if class_total[key][i] > 0:
                accuracy = 100 * class_correct[key][i] / class_total[key][i]
                print(f"Class {i}: {accuracy:.2f}%")
            else:
                print(f"Class {i}: No samples found.")


# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
main_optimizer = optim.Adam(main_model.parameters(), lr=0.001)

# Train LeNet
print("Training the main model (LeNet)...")
train_lenet(main_model, device, train_loader, main_optimizer, criterion, epochs=10)

# Define and train auxiliary models for convolutional layers
auxiliary_models = {
    'conv1': AuxiliaryModel(6 * 24 * 24).to(device),
    'conv2': AuxiliaryModel(16 * 8 * 8).to(device),
}
aux_optimizers = {
    layer: optim.Adam(auxiliary_models[layer].parameters(), lr=0.001)
    for layer in auxiliary_models.keys()
}
aux_criteria = {
    layer: nn.CrossEntropyLoss()
    for layer in auxiliary_models.keys()
}

print("Training auxiliary models...")
train_auxiliary_models(main_model, auxiliary_models, device, train_loader, aux_optimizers, aux_criteria, epochs=10)

# Evaluate the main model and auxiliary models
print("Evaluating models...")
evaluate_models(main_model, auxiliary_models, device, test_loader)
