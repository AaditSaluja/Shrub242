import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for AlexNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        self.layer_outputs = {}
        # print(f"Input shape: {x.shape}")  # Debug input shape
        x = self.features[0](x)
        self.layer_outputs['conv1'] = x
        # print(f"conv1 shape: {x.shape}")  # Debug conv1 output shape
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        self.layer_outputs['conv2'] = x
        # print(f"conv2 shape: {x.shape}")  # Debug conv2 output shape
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        self.layer_outputs['conv3'] = x
        # print(f"conv3 shape: {x.shape}")  # Debug conv3 output shape
        x = self.features[7](x)
        x = self.features[8](x)
        self.layer_outputs['conv4'] = x
        # print(f"conv4 shape: {x.shape}")  # Debug conv4 output shape
        x = self.features[9](x)
        x = self.features[10](x)
        self.layer_outputs['conv5'] = x
        # print(f"conv5 shape: {x.shape}")  # Debug conv5 output shape
        x = self.features[11](x)
        x = self.features[12](x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define an auxiliary model for each convolutional layer
class AuxiliaryModel(nn.Module):
    def __init__(self, input_dim):
        super(AuxiliaryModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(x)

# Training function for the main model (AlexNet)
def train_alexnet(model, device, train_loader, optimizer, criterion, epochs=50):
    model.train()
    start_time = time.time()  # Start timing

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

    elapsed_time = time.time() - start_time  # End timing
    print(f"Time taken to train main model: {elapsed_time:.2f} seconds")

# Training function for auxiliary models
def train_auxiliary_models(model, auxiliary_models, device, train_loader, optimizers, criteria, epochs=20):
    for param in model.parameters():
        param.requires_grad = False  # Freeze AlexNet weights

    model.eval()  # AlexNet remains in evaluation mode
    start_time = time.time() 
    for epoch in range(epochs):
        running_losses = {layer: 0.0 for layer in auxiliary_models.keys()}
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            _ = model(images)  # Forward pass to populate layer_outputs

            # Train auxiliary models
            # Inside train_auxiliary_models
            for layer_name, aux_model in auxiliary_models.items():
                aux_optimizer = optimizers[layer_name]
                aux_criterion = criteria[layer_name]

                # Detach and flatten the layer output
                layer_output = model.layer_outputs[layer_name].detach()
                layer_output_flattened = torch.flatten(layer_output, start_dim=1)  # Flatten the output

                # Debug: Print shapes
                # print(f"{layer_name} flattened output shape: {layer_output_flattened.shape}")
                # print(f"{layer_name} auxiliary model input size: {aux_model.fc[1].in_features}")

                aux_output = aux_model(layer_output_flattened)
                aux_loss = aux_criterion(aux_output, labels)

                aux_optimizer.zero_grad()
                aux_loss.backward()
                aux_optimizer.step()

                running_losses[layer_name] += aux_loss.item()


        print(f"Epoch {epoch + 1}, " +
              ", ".join([f"{layer} Loss: {running_losses[layer] / len(train_loader):.4f}" for layer in auxiliary_models.keys()]))
    
    elapsed_time = time.time() - start_time  # End timing
    print(f"Time taken to train auxiliary models: {elapsed_time:.2f} seconds")

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
    
    num_classes = 10  # CIFAR-10 has 10 classes
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
                print(f"Class {i} ({test_dataset.classes[i]}): {accuracy:.2f}%")
            else:
                print(f"Class {i} ({test_dataset.classes[i]}): No samples found.")


# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
main_optimizer = optim.Adam(main_model.parameters(), lr=0.001)

# Train AlexNet
print("Training the main model (AlexNet)...")
train_alexnet(main_model, device, train_loader, main_optimizer, criterion, epochs=50)

# Define and train auxiliary models for convolutional layers
auxiliary_models = {
    'conv1': AuxiliaryModel(64 * 55 * 55).to(device),  # 193600
    'conv2': AuxiliaryModel(192 * 27 * 27).to(device),  # 139968
    'conv3': AuxiliaryModel(384 * 13 * 13).to(device),  # 64768
    'conv4': AuxiliaryModel(256 * 13 * 13).to(device),  # 43264
    'conv5': AuxiliaryModel(256 * 13 * 13).to(device),    # 9216
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
train_auxiliary_models(main_model, auxiliary_models, device, train_loader, aux_optimizers, aux_criteria, epochs=20)

# Evaluate the main model and auxiliary models
print("Evaluating models...")
evaluate_models(main_model, auxiliary_models, device, test_loader)
