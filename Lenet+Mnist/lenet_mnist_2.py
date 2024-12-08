# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # Load and preprocess the MNIST dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # class LeNetWithAuxiliary(nn.Module):
# #     def __init__(self):
# #         super(LeNetWithAuxiliary, self).__init__()
# #         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Input: 1x28x28, Output: 6x24x24
# #         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 6x12x12
# #         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output: 16x8x8
# #         self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten to 16x4x4
# #         self.fc2 = nn.Linear(120, 84)
# #         self.fc3 = nn.Linear(84, 10)

# #         # Auxiliary classifiers
# #         self.aux1 = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(6 * 24 * 24, 10)  # Corrected size for conv1
# #         )
# #         self.aux2 = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(16 * 4 * 4, 10)  # Corrected size for pool2
# #         )
# #         self.aux3 = nn.Sequential(
# #             nn.Linear(120, 10)
# #         )
# #         self.aux4 = nn.Sequential(
# #             nn.Linear(84, 10)
# #         )

# #     def forward(self, x):
# #         self.layer_outputs = {}
        
# #         # First convolution and auxiliary
# #         x = torch.relu(self.conv1(x))
# #         self.layer_outputs['conv1'] = x
# #         aux1_output = self.aux1(self.layer_outputs['conv1'])

# #         # First pooling
# #         x = self.pool(x)
# #         self.layer_outputs['pool1'] = x

# #         # Second convolution
# #         x = torch.relu(self.conv2(x))
# #         self.layer_outputs['conv2'] = x

# #         # Second pooling and auxiliary
# #         x = self.pool(x)
# #         self.layer_outputs['pool2'] = x
# #         aux2_output = self.aux2(self.layer_outputs['pool2'])

# #         # Fully connected layers
# #         x = x.view(-1, 16 * 4 * 4)
# #         x = torch.relu(self.fc1(x))
# #         self.layer_outputs['fc1'] = x
# #         aux3_output = self.aux3(self.layer_outputs['fc1'])

# #         x = torch.relu(self.fc2(x))
# #         self.layer_outputs['fc2'] = x
# #         aux4_output = self.aux4(self.layer_outputs['fc2'])

# #         x = self.fc3(x)
# #         return x, aux1_output, aux2_output, aux3_output, aux4_output

# class LeNetWithDirectAuxiliary(nn.Module):
#     def __init__(self):
#         super(LeNetWithDirectAuxiliary, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Input: 1x28x28, Output: 6x24x24
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 6x12x12
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output: 16x8x8
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten to 16x4x4
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         self.layer_outputs = {}

#         # First convolution
#         x = torch.relu(self.conv1(x))
#         self.layer_outputs['conv1'] = x  # Store the output for classification

#         # First pooling
#         x = self.pool(x)
#         self.layer_outputs['pool1'] = x  # Store the output for classification

#         # Second convolution
#         x = torch.relu(self.conv2(x))
#         self.layer_outputs['conv2'] = x  # Store the output for classification

#         # Second pooling
#         x = self.pool(x)
#         self.layer_outputs['pool2'] = x  # Store the output for classification

#         # Fully connected layers
#         x = x.view(-1, 16 * 4 * 4)
#         x = torch.relu(self.fc1(x))
#         self.layer_outputs['fc1'] = x  # Store the output for classification

#         x = torch.relu(self.fc2(x))
#         self.layer_outputs['fc2'] = x  # Store the output for classification

#         x = self.fc3(x)
#         return x


# def train_model(model, device, train_loader, optimizer, criterion, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)  # Only the final output is returned
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")


# # Train the model with auxiliary losses
# # def train_model_with_auxiliary(model, device, train_loader, optimizer, criterion, epochs=5):
# #     model.train()
# #     for epoch in range(epochs):
# #         running_loss = 0.0
# #         for images, labels in train_loader:
# #             images, labels = images.to(device), labels.to(device)

# #             optimizer.zero_grad()
# #             outputs, aux1_output, aux2_output, aux3_output, aux4_output = model(images)

# #             # Main loss and auxiliary losses
# #             main_loss = criterion(outputs, labels)
# #             aux1_loss = criterion(aux1_output, labels)
# #             aux2_loss = criterion(aux2_output, labels)
# #             aux3_loss = criterion(aux3_output, labels)
# #             aux4_loss = criterion(aux4_output, labels)

# #             total_loss = main_loss + aux1_loss + aux2_loss + aux3_loss + aux4_loss
# #             total_loss.backward()
# #             optimizer.step()

# #             running_loss += total_loss.item()
# #         print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# def classify_with_layers(model, images, device):
#     layer_outputs = model(images.to(device))

#     # Classification outputs
#     classifications = {}
#     for layer_name, output in model.layer_outputs.items():
#         if len(output.shape) == 4:  # Convolutional layer outputs
#             output = output.mean(dim=[2, 3])  # Global average pooling
#         classifications[layer_name] = torch.argmax(output, dim=1)
#     return classifications


# # Evaluate the model and auxiliary classifiers
# # def evaluate_model_with_auxiliary(model, device, test_loader):
# #     model.eval()
# #     correct = {'main': 0, 'aux1': 0, 'aux2': 0, 'aux3': 0, 'aux4': 0}
# #     total = 0

# #     with torch.no_grad():
# #         for images, labels in test_loader:
# #             images, labels = images.to(device), labels.to(device)
# #             outputs, aux1_output, aux2_output, aux3_output, aux4_output = model(images)

# #             predictions = {
# #                 'main': torch.argmax(outputs, 1),
# #                 'aux1': torch.argmax(aux1_output, 1),
# #                 'aux2': torch.argmax(aux2_output, 1),
# #                 'aux3': torch.argmax(aux3_output, 1),
# #                 'aux4': torch.argmax(aux4_output, 1),
# #             }

# #             for key in correct.keys():
# #                 correct[key] += (predictions[key] == labels).sum().item()
# #             total += labels.size(0)

# #     for key, num_correct in correct.items():
# #         print(f"Accuracy for {key}: {100 * num_correct / total:.2f}%")

# def evaluate_model_with_direct_layers(model, device, test_loader):
#     model.eval()
#     correct = {layer_name: 0 for layer_name in model.layer_outputs.keys()}
#     total = 0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             _ = model(images)  # Forward pass to compute layer outputs

#             for layer_name, output in model.layer_outputs.items():
#                 if len(output.shape) == 4:  # Convolutional layers
#                     output = output.mean(dim=[2, 3])  # Global average pooling
#                 preds = torch.argmax(output, dim=1)
#                 correct[layer_name] += (preds == labels).sum().item()
#             total += labels.size(0)

#     for layer_name, num_correct in correct.items():
#         print(f"Accuracy for {layer_name}: {100 * num_correct / total:.2f}%")


# # Main script
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LeNetWithDirectAuxiliary().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# print("Training the model with auxiliary classifiers...")
# train_model(model, device, train_loader, optimizer, criterion, epochs=5)

# print("Evaluating the model...")
# evaluate_model_with_direct_layers(model, device, test_loader)


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

# Define the LeNet model with auxiliary classifiers
class LeNetWithAuxiliary(nn.Module):
    def __init__(self):
        super(LeNetWithAuxiliary, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Input: 1x28x28, Output: 6x24x24
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 6x12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output: 16x8x8
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten to 16x4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Auxiliary classifiers
        self.aux1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 24 * 24, 10)  # For conv1 output
        )
        self.aux2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 10)  # For pool2 output
        )

    def forward(self, x):
        self.layer_outputs = {}
        
        # First convolution
        x = torch.relu(self.conv1(x))
        self.layer_outputs['conv1'] = x
        aux1_output = self.aux1(x)

        # First pooling
        x = self.pool(x)
        self.layer_outputs['pool1'] = x

        # Second convolution
        x = torch.relu(self.conv2(x))
        self.layer_outputs['conv2'] = x

        # Second pooling
        x = self.pool(x)
        self.layer_outputs['pool2'] = x
        aux2_output = self.aux2(x)

        # Fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        self.layer_outputs['fc1'] = x

        x = torch.relu(self.fc2(x))
        self.layer_outputs['fc2'] = x

        x = self.fc3(x)
        return x, aux1_output, aux2_output

# Training function with auxiliary losses
def train_model_with_auxiliary(model, device, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, aux1_output, aux2_output = model(images)

            # Main loss and auxiliary losses
            main_loss = criterion(outputs, labels)
            aux1_loss = criterion(aux1_output, labels)
            aux2_loss = criterion(aux2_output, labels)
            total_loss = main_loss + aux1_loss + aux2_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation function with auxiliary classifiers
def evaluate_model_with_auxiliary(model, device, test_loader):
    model.eval()
    correct = {'main': 0, 'aux1': 0, 'aux2': 0}
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, aux1_output, aux2_output = model(images)

            predictions = {
                'main': torch.argmax(outputs, 1),
                'aux1': torch.argmax(aux1_output, 1),
                'aux2': torch.argmax(aux2_output, 1),
            }

            for key in correct.keys():
                correct[key] += (predictions[key] == labels).sum().item()
            total += labels.size(0)

    for key, num_correct in correct.items():
        print(f"Accuracy for {key}: {100 * num_correct / total:.2f}%")

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNetWithAuxiliary().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model with auxiliary classifiers...")
train_model_with_auxiliary(model, device, train_loader, optimizer, criterion, epochs=5)

print("Evaluating the model...")
evaluate_model_with_auxiliary(model, device, test_loader)
