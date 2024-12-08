# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torchvision import datasets, transforms, models
# # from torch.utils.data import DataLoader

# # # Device configuration
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Data directory (replace with your ImageNet data directory)
# # data_dir = '/path/to/imagenet/'

# # # Data transformations for ImageNet
# # transform = transforms.Compose([
# #     transforms.Resize(256),
# #     transforms.CenterCrop(224),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
# #                          std=[0.229, 0.224, 0.225])   # ImageNet stds
# # ])

# # # Load the ImageNet dataset
# # train_dataset = datasets.ImageNet(root=data_dir, split='train', transform=transform)
# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# # val_dataset = datasets.ImageNet(root=data_dir, split='val', transform=transform)
# # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# # # Load pre-trained ResNet18
# # main_model = models.resnet18(pretrained=True).to(device)

# # # Dictionary to store outputs from convolutional layers
# # layer_outputs = {}

# # # Function to save outputs from layers
# # def get_activation(name):
# #     def hook(model, input, output):
# #         layer_outputs[name] = output.detach()
# #     return hook

# # # Register hooks to capture outputs from all convolutional layers
# # layer_names = []
# # layers_to_hook = []

# # # First convolutional layer
# # layer_names.append('conv1')
# # layers_to_hook.append(main_model.conv1)

# # # Convolutional layers in the residual blocks
# # for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
# #     layer = getattr(main_model, layer_name)
# #     for block_idx, block in enumerate(layer):
# #         for conv_name in ['conv1', 'conv2']:
# #             conv = getattr(block, conv_name)
# #             name = f"{layer_name}.{block_idx}.{conv_name}"
# #             layer_names.append(name)
# #             layers_to_hook.append(conv)

# # # Register hooks
# # for name, layer in zip(layer_names, layers_to_hook):
# #     layer.register_forward_hook(get_activation(name))

# # # Define the auxiliary model for each convolutional layer
# # class AuxiliaryModel(nn.Module):
# #     def __init__(self, input_dim, num_classes=1000):
# #         super(AuxiliaryModel, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(input_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, num_classes)
# #         )

# #     def forward(self, x):
# #         return self.fc(x)

# # # Initialize auxiliary models with correct input dimensions
# # auxiliary_models = {}

# # # Pass a batch to get the output shapes
# # with torch.no_grad():
# #     images, labels = next(iter(train_loader))
# #     images = images.to(device)
# #     _ = main_model(images)

# # for name in layer_names:
# #     output = layer_outputs[name]
# #     num_features = output.view(output.size(0), -1).size(1)
# #     aux_model = AuxiliaryModel(num_features, num_classes=1000).to(device)
# #     auxiliary_models[name] = aux_model

# # # Define optimizers and loss functions for auxiliary models
# # aux_optimizers = {
# #     layer: optim.Adam(auxiliary_models[layer].parameters(), lr=0.001)
# #     for layer in auxiliary_models.keys()
# # }

# # aux_criteria = {
# #     layer: nn.CrossEntropyLoss()
# #     for layer in auxiliary_models.keys()
# # }

# # # Training function for auxiliary models
# # def train_auxiliary_models(main_model, auxiliary_models, device, train_loader, optimizers, criteria, epochs=3):
# #     for param in main_model.parameters():
# #         param.requires_grad = False  # Freeze main model weights

# #     main_model.eval()  # Main model remains in evaluation mode

# #     for epoch in range(epochs):
# #         running_losses = {layer: 0.0 for layer in auxiliary_models.keys()}
# #         for images, labels in train_loader:
# #             images, labels = images.to(device), labels.to(device)

# #             layer_outputs.clear()  # Clear previous outputs

# #             _ = main_model(images)  # Forward pass to populate layer_outputs

# #             # Train auxiliary models
# #             for layer_name, aux_model in auxiliary_models.items():
# #                 aux_optimizer = optimizers[layer_name]
# #                 aux_criterion = criteria[layer_name]

# #                 layer_output = layer_outputs[layer_name]
# #                 aux_output = aux_model(layer_output)
# #                 aux_loss = aux_criterion(aux_output, labels)

# #                 aux_optimizer.zero_grad()
# #                 aux_loss.backward()
# #                 aux_optimizer.step()

# #                 running_losses[layer_name] += aux_loss.item()

# #         # Print average losses
# #         print(f"Epoch {epoch + 1}, " +
# #               ", ".join([f"{layer} Loss: {running_losses[layer] / len(train_loader):.4f}" for layer in auxiliary_models.keys()]))

# # # Evaluation function for main and auxiliary models
# # def evaluate_models(main_model, auxiliary_models, device, test_loader):
# #     main_model.eval()
# #     for aux_model in auxiliary_models.values():
# #         aux_model.eval()

# #     correct = {layer: 0 for layer in ['main'] + list(auxiliary_models.keys())}
# #     total = 0

# #     with torch.no_grad():
# #         for images, labels in test_loader:
# #             images, labels = images.to(device), labels.to(device)
# #             outputs = main_model(images)

# #             # Main model predictions
# #             _, predicted = torch.max(outputs, 1)
# #             correct['main'] += (predicted == labels).sum().item()
# #             total += labels.size(0)

# #             # Auxiliary models predictions
# #             for layer_name, aux_model in auxiliary_models.items():
# #                 layer_output = layer_outputs[layer_name]
# #                 aux_output = aux_model(layer_output)
# #                 _, predicted = torch.max(aux_output, 1)
# #                 correct[layer_name] += (predicted == labels).sum().item()

# #     # Print overall accuracies
# #     for key in correct.keys():
# #         accuracy = 100 * correct[key] / total
# #         print(f"{key} Accuracy: {accuracy:.2f}%")

# # # Main script
# # print("Training auxiliary models...")
# # train_auxiliary_models(main_model, auxiliary_models, device, train_loader, aux_optimizers, aux_criteria, epochs=3)

# # # Evaluate the main model and auxiliary models
# # print("Evaluating models...")
# # evaluate_models(main_model, auxiliary_models, device, val_loader)


# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torchvision import datasets, transforms, models
# # from torch.utils.data import DataLoader

# # # Device configuration
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Data transformations for CIFAR-100
# # transform = transforms.Compose([
# #     transforms.Resize(224),  # ResNet18 expects 224x224 images
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 means
# #                          std=[0.2675, 0.2565, 0.2761])   # CIFAR-100 stds
# # ])

# # # Load the CIFAR-100 dataset
# # train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# # val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
# # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# # num_classes = 100  # CIFAR-100 has 100 classes

# # # Load pre-trained ResNet18
# # main_model = models.resnet18(pretrained=True).to(device)

# # # Modify the final fully connected layer to match CIFAR-100 classes
# # main_model.fc = nn.Linear(main_model.fc.in_features, num_classes).to(device)

# # # Dictionary to store outputs from convolutional layers
# # layer_outputs = {}

# # # Function to save outputs from layers
# # def get_activation(name):
# #     def hook(model, input, output):
# #         layer_outputs[name] = output.detach()
# #     return hook

# # # Register hooks to capture outputs from all convolutional layers
# # layer_names = []
# # layers_to_hook = []

# # # First convolutional layer
# # layer_names.append('conv1')
# # layers_to_hook.append(main_model.conv1)

# # # Convolutional layers in the residual blocks
# # for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
# #     layer = getattr(main_model, layer_name)
# #     for block_idx, block in enumerate(layer):
# #         for conv_name in ['conv1', 'conv2']:
# #             conv = getattr(block, conv_name)
# #             name = f"{layer_name}.{block_idx}.{conv_name}"
# #             layer_names.append(name)
# #             layers_to_hook.append(conv)

# # # Register hooks
# # for name, layer in zip(layer_names, layers_to_hook):
# #     layer.register_forward_hook(get_activation(name))

# # # Define the auxiliary model for each convolutional layer
# # class AuxiliaryModel(nn.Module):
# #     def __init__(self, input_dim, num_classes):
# #         super(AuxiliaryModel, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(input_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, num_classes)
# #         )

# #     def forward(self, x):
# #         return self.fc(x)

# # # Initialize auxiliary models with correct input dimensions
# # auxiliary_models = {}

# # # Pass a batch to get the output shapes
# # with torch.no_grad():
# #     images, labels = next(iter(train_loader))
# #     images = images.to(device)
# #     _ = main_model(images)

# # for name in layer_names:
# #     output = layer_outputs[name]
# #     num_features = output.view(output.size(0), -1).size(1)
# #     aux_model = AuxiliaryModel(num_features, num_classes=num_classes).to(device)
# #     auxiliary_models[name] = aux_model

# # # Define optimizers and loss functions for auxiliary models
# # aux_optimizers = {
# #     layer: optim.Adam(auxiliary_models[layer].parameters(), lr=0.001)
# #     for layer in auxiliary_models.keys()
# # }

# # aux_criteria = {
# #     layer: nn.CrossEntropyLoss()
# #     for layer in auxiliary_models.keys()
# # }

# # # Training function for auxiliary models
# # def train_auxiliary_models(main_model, auxiliary_models, device, train_loader, optimizers, criteria, epochs=3):
# #     for param in main_model.parameters():
# #         param.requires_grad = False  # Freeze main model weights

# #     main_model.eval()  # Main model remains in evaluation mode

# #     for epoch in range(epochs):
# #         running_losses = {layer: 0.0 for layer in auxiliary_models.keys()}
# #         for images, labels in train_loader:
# #             images, labels = images.to(device), labels.to(device)

# #             layer_outputs.clear()  # Clear previous outputs

# #             _ = main_model(images)  # Forward pass to populate layer_outputs

# #             # Train auxiliary models
# #             for layer_name, aux_model in auxiliary_models.items():
# #                 aux_optimizer = optimizers[layer_name]
# #                 aux_criterion = criteria[layer_name]

# #                 layer_output = layer_outputs[layer_name]
# #                 aux_output = aux_model(layer_output)
# #                 aux_loss = aux_criterion(aux_output, labels)

# #                 aux_optimizer.zero_grad()
# #                 aux_loss.backward()
# #                 aux_optimizer.step()

# #                 running_losses[layer_name] += aux_loss.item()

# #         # Print average losses for the epoch
# #         print(f"Epoch {epoch + 1}, " +
# #               ", ".join([f"{layer} Loss: {running_losses[layer] / len(train_loader):.4f}" for layer in auxiliary_models.keys()]))

# # # Evaluation function for main and auxiliary models
# # def evaluate_models(main_model, auxiliary_models, device, test_loader):
# #     main_model.eval()
# #     for aux_model in auxiliary_models.values():
# #         aux_model.eval()

# #     correct = {layer: 0 for layer in ['main'] + list(auxiliary_models.keys())}
# #     total = 0

# #     with torch.no_grad():
# #         for images, labels in test_loader:
# #             images, labels = images.to(device), labels.to(device)
# #             outputs = main_model(images)

# #             # Main model predictions
# #             _, predicted = torch.max(outputs, 1)
# #             correct['main'] += (predicted == labels).sum().item()
# #             total += labels.size(0)

# #             # Auxiliary models predictions
# #             for layer_name, aux_model in auxiliary_models.items():
# #                 layer_output = layer_outputs[layer_name]
# #                 aux_output = aux_model(layer_output)
# #                 _, predicted = torch.max(aux_output, 1)
# #                 correct[layer_name] += (predicted == labels).sum().item()

# #     # Print overall accuracies
# #     for key in correct.keys():
# #         accuracy = 100 * correct[key] / total
# #         print(f"{key} Accuracy: {accuracy:.2f}%")

# # # Main script
# # print("Training auxiliary models...")
# # train_auxiliary_models(main_model, auxiliary_models, device, train_loader, aux_optimizers, aux_criteria, epochs=3)

# # # Evaluate the main model and auxiliary models
# # print("Evaluating models...")
# # evaluate_models(main_model, auxiliary_models, device, val_loader)


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Data transformations for CIFAR-100
# transform = transforms.Compose([
#     transforms.Resize(224),  # ResNet18 expects 224x224 images
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 means
#                          std=[0.2675, 0.2565, 0.2761])   # CIFAR-100 stds
# ])

# # Load the CIFAR-100 dataset
# train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# num_classes = 100  # CIFAR-100 has 100 classes

# # Load pre-trained ResNet18
# main_model = models.resnet18(pretrained=True).to(device)

# # Modify the final fully connected layer to match CIFAR-100 classes
# main_model.fc = nn.Linear(main_model.fc.in_features, num_classes).to(device)

# # Dictionary to store outputs from convolutional layers
# layer_outputs = {}

# # Function to save outputs from layers
# def get_activation(name):
#     def hook(model, input, output):
#         layer_outputs[name] = output.detach()
#     return hook

# # Register hooks to capture outputs from all convolutional layers
# layer_names = []
# layers_to_hook = []

# # First convolutional layer
# layer_names.append('conv1')
# layers_to_hook.append(main_model.conv1)

# # Convolutional layers in the residual blocks
# for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
#     layer = getattr(main_model, layer_name)
#     for block_idx, block in enumerate(layer):
#         for conv_name in ['conv1', 'conv2']:
#             conv = getattr(block, conv_name)
#             name = f"{layer_name}.{block_idx}.{conv_name}"
#             layer_names.append(name)
#             layers_to_hook.append(conv)

# # Register hooks
# for name, layer in zip(layer_names, layers_to_hook):
#     layer.register_forward_hook(get_activation(name))

# # Define the auxiliary model for each convolutional layer
# class AuxiliaryModel(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(AuxiliaryModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         return self.fc(x)

# # Initialize auxiliary models with correct input dimensions
# auxiliary_models = {}

# # Pass a batch to get the output shapes
# with torch.no_grad():
#     images, labels = next(iter(train_loader))
#     images = images.to(device)
#     _ = main_model(images)

# for name in layer_names:
#     output = layer_outputs[name]
#     num_features = output.view(output.size(0), -1).size(1)
#     aux_model = AuxiliaryModel(num_features, num_classes=num_classes).to(device)
#     auxiliary_models[name] = aux_model

# # Define optimizers and loss functions for auxiliary models
# aux_optimizers = {
#     layer: optim.Adam(auxiliary_models[layer].parameters(), lr=0.001)
#     for layer in auxiliary_models.keys()
# }

# aux_criteria = {
#     layer: nn.CrossEntropyLoss()
#     for layer in auxiliary_models.keys()
# }

# # Training function for the main model
# def train_main_model(model, device, train_loader, optimizer, criterion, epochs=100):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             # Calculate accuracy on the training batch
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         accuracy = 100 * correct / total
#         print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {accuracy:.2f}%")

# # Training function for auxiliary models
# def train_auxiliary_models(main_model, auxiliary_models, device, train_loader, optimizers, criteria, epochs=100):
#     for param in main_model.parameters():
#         param.requires_grad = False  # Freeze main model weights

#     main_model.eval()  # Main model remains in evaluation mode

#     for epoch in range(epochs):
#         running_losses = {layer: 0.0 for layer in auxiliary_models.keys()}
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             layer_outputs.clear()  # Clear previous outputs

#             _ = main_model(images)  # Forward pass to populate layer_outputs

#             # Train auxiliary models
#             for layer_name, aux_model in auxiliary_models.items():
#                 aux_optimizer = optimizers[layer_name]
#                 aux_criterion = criteria[layer_name]

#                 layer_output = layer_outputs[layer_name]
#                 aux_output = aux_model(layer_output)
#                 aux_loss = aux_criterion(aux_output, labels)

#                 aux_optimizer.zero_grad()
#                 aux_loss.backward()
#                 aux_optimizer.step()

#                 running_losses[layer_name] += aux_loss.item()

#         # Print average losses for the epoch
#         print(f"Epoch {epoch + 1}, " +
#               ", ".join([f"{layer} Loss: {running_losses[layer] / len(train_loader):.4f}" for layer in auxiliary_models.keys()]))

# # Evaluation function for main and auxiliary models
# def evaluate_models(main_model, auxiliary_models, device, test_loader):
#     main_model.eval()
#     for aux_model in auxiliary_models.values():
#         aux_model.eval()

#     correct = {layer: 0 for layer in ['main'] + list(auxiliary_models.keys())}
#     total = 0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = main_model(images)

#             # Main model predictions
#             _, predicted = torch.max(outputs, 1)
#             correct['main'] += (predicted == labels).sum().item()
#             total += labels.size(0)

#             # Auxiliary models predictions
#             for layer_name, aux_model in auxiliary_models.items():
#                 layer_output = layer_outputs[layer_name]
#                 aux_output = aux_model(layer_output)
#                 _, predicted = torch.max(aux_output, 1)
#                 correct[layer_name] += (predicted == labels).sum().item()

#     # Print overall accuracies
#     for key in correct.keys():
#         accuracy = 100 * correct[key] / total
#         print(f"{key} Accuracy: {accuracy:.2f}%")

# # Main script
# # Step 1: Train the main model
# print("Training the main model...")

# # Unfreeze all layers or just the final layer depending on your choice
# # Option 1: Train only the final layer (feature extraction)
# for param in main_model.parameters():
#     param.requires_grad = False  # Freeze all layers

# for param in main_model.fc.parameters():
#     param.requires_grad = True  # Unfreeze final layer

# # Option 2: Fine-tune the entire model
# # for param in main_model.parameters():
# #     param.requires_grad = True  # Unfreeze all layers

# # Set the model to training mode
# main_model.train()

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(main_model.fc.parameters(), lr=0.001)  # Use main_model.parameters() if fine-tuning

# # Train the main model
# train_main_model(main_model, device, train_loader, optimizer, criterion, epochs=100)

# # Step 2: Train auxiliary models
# print("Training auxiliary models...")
# train_auxiliary_models(main_model, auxiliary_models, device, train_loader, aux_optimizers, aux_criteria, epochs=100)

# # Evaluate the main model and auxiliary models
# print("Evaluating models...")
# evaluate_models(main_model, auxiliary_models, device, val_loader)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations for CIFAR-100
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet18 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 means
                         std=[0.2675, 0.2565, 0.2761])   # CIFAR-100 stds
])

# Load the CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

num_classes = 100  # CIFAR-100 has 100 classes

# Load pre-trained ResNet18
main_model = models.resnet18(pretrained=True).to(device)

# Modify the final fully connected layer to match CIFAR-100 classes
main_model.fc = nn.Linear(main_model.fc.in_features, num_classes).to(device)

# Dictionary to store outputs from convolutional layers
layer_outputs = {}

# Function to save outputs from layers
def get_activation(name):
    def hook(model, input, output):
        layer_outputs[name] = output.detach()
    return hook

# Register hooks to capture outputs from all convolutional layers
layer_names = []
layers_to_hook = []

# First convolutional layer
layer_names.append('conv1')
layers_to_hook.append(main_model.conv1)

# Convolutional layers in the residual blocks
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    layer = getattr(main_model, layer_name)
    for block_idx, block in enumerate(layer):
        for conv_name in ['conv1', 'conv2']:
            conv = getattr(block, conv_name)
            name = f"{layer_name}.{block_idx}.{conv_name}"
            layer_names.append(name)
            layers_to_hook.append(conv)

# Register hooks
for name, layer in zip(layer_names, layers_to_hook):
    layer.register_forward_hook(get_activation(name))

# Define the auxiliary model for each convolutional layer
class AuxiliaryModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(AuxiliaryModel, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize auxiliary models with correct input dimensions
auxiliary_models = {}

# Pass a batch to get the output shapes
with torch.no_grad():
    images, labels = next(iter(train_loader))
    images = images.to(device)
    layer_outputs.clear()  # Clear any existing outputs
    _ = main_model(images)

for name in layer_names:
    output = layer_outputs[name]
    num_channels = output.size(1)
    aux_model = AuxiliaryModel(num_channels, num_classes=num_classes).to(device)
    auxiliary_models[name] = aux_model

# Define optimizers and loss functions for auxiliary models
aux_optimizers = {
    layer: optim.Adam(auxiliary_models[layer].parameters(), lr=0.001)
    for layer in auxiliary_models.keys()
}

aux_criteria = {
    layer: nn.CrossEntropyLoss()
    for layer in auxiliary_models.keys()
}

# Training function for the main model
def train_main_model(model, device, train_loader, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy on the training batch
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {accuracy:.2f}%")

# Training function for auxiliary models
def train_auxiliary_models(main_model, auxiliary_models, device, train_loader, optimizers, criteria, epochs=100):
    for param in main_model.parameters():
        param.requires_grad = False  # Freeze main model weights

    main_model.eval()  # Main model remains in evaluation mode

    for epoch in range(epochs):
        running_losses = {layer: 0.0 for layer in auxiliary_models.keys()}
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            layer_outputs.clear()  # Clear previous outputs
            _ = main_model(images)  # Forward pass to populate layer_outputs

            # Train auxiliary models
            for layer_name, aux_model in auxiliary_models.items():
                aux_optimizer = optimizers[layer_name]
                aux_criterion = criteria[layer_name]

                layer_output = layer_outputs[layer_name]
                aux_output = aux_model(layer_output)
                aux_loss = aux_criterion(aux_output, labels)

                aux_optimizer.zero_grad()
                aux_loss.backward()
                aux_optimizer.step()

                running_losses[layer_name] += aux_loss.item()

        # Print average losses for the epoch
        print(f"Epoch {epoch + 1}, " +
              ", ".join([f"{layer} Loss: {running_losses[layer] / len(train_loader):.4f}" for layer in auxiliary_models.keys()]))

# Evaluation function for main and auxiliary models
def evaluate_models(main_model, auxiliary_models, device, test_loader):
    main_model.eval()
    for aux_model in auxiliary_models.values():
        aux_model.eval()

    correct = {layer: 0 for layer in ['main'] + list(auxiliary_models.keys())}
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            layer_outputs.clear()  # Clear previous outputs
            outputs = main_model(images)

            # Main model predictions
            _, predicted = torch.max(outputs, 1)
            correct['main'] += (predicted == labels).sum().item()
            total += labels.size(0)

            # Auxiliary models predictions
            for layer_name, aux_model in auxiliary_models.items():
                layer_output = layer_outputs[layer_name]
                aux_output = aux_model(layer_output)
                _, predicted = torch.max(aux_output, 1)
                correct[layer_name] += (predicted == labels).sum().item()

    # Print overall accuracies
    for key in correct.keys():
        accuracy = 100 * correct[key] / total
        print(f"{key} Accuracy: {accuracy:.2f}%")

# Main script
# Step 1: Train the main model
print("Training the main model...")

# Unfreeze all layers or just the final layer depending on your choice
# Option 1: Train only the final layer (feature extraction)
for param in main_model.parameters():
    param.requires_grad = False  # Freeze all layers

for param in main_model.fc.parameters():
    param.requires_grad = True  # Unfreeze final layer

# Option 2: Fine-tune the entire model
# Uncomment the following lines if you wish to fine-tune the entire model
# for param in main_model.parameters():
#     param.requires_grad = True  # Unfreeze all layers

# Set the model to training mode
main_model.train()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(main_model.fc.parameters(), lr=0.001)  # Use main_model.parameters() if fine-tuning

# Train the main model
train_main_model(main_model, device, train_loader, optimizer, criterion, epochs=25)  # Adjust epochs as needed

# Step 2: Train auxiliary models
print("Training auxiliary models...")
train_auxiliary_models(main_model, auxiliary_models, device, train_loader, aux_optimizers, aux_criteria, epochs=25)  # Adjust epochs as needed

# Evaluate the main model and auxiliary models
print("Evaluating models...")
evaluate_models(main_model, auxiliary_models, device, val_loader)
