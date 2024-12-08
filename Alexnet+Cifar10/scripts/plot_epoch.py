import matplotlib.pyplot as plt 
# Treating x-axis as text labels
epochs_labels = ["Epoch 1", "Epoch 5", "Epoch 15", "Epoch 25"]

main_color = "black"
conv1_color = "orange"
conv2_color = "green"
conv3_color = "red"
conv4_color = "purple"
conv5_color = "yellow"

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_labels, main, label="Main", marker='o', color=main_color)
plt.plot(epochs_labels, conv1, label="Conv1", marker='o', color=conv1_color)
plt.plot(epochs_labels, conv2, label="Conv2", marker='o', color=conv2_color)
plt.plot(epochs_labels, conv3, label="Conv3", marker='o', color=conv3_color)
plt.plot(epochs_labels, conv4, label="Conv4", marker='o', color=conv4_color)
plt.plot(epochs_labels, conv5, label="Conv5", marker='o', color=conv5_color)

# Customization
plt.title("AlexNet with CIFAR-10", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.legend(title="Layers")
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
