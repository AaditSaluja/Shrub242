# Define the accuracies and depths of each layer
layer_data = [
    {'name': 'conv1', 'accuracy': 12.16, 'depth': 1},
    {'name': 'layer1.0.conv1', 'accuracy': 16.07, 'depth': 2},
    {'name': 'layer1.0.conv2', 'accuracy': 21.14, 'depth': 3},
    {'name': 'layer1.1.conv1', 'accuracy': 19.68, 'depth': 4},
    {'name': 'layer1.1.conv2', 'accuracy': 21.25, 'depth': 5},
    {'name': 'layer2.0.conv1', 'accuracy': 24.70, 'depth': 6},
    {'name': 'layer2.0.conv2', 'accuracy': 23.57, 'depth': 7},
    {'name': 'layer2.1.conv1', 'accuracy': 25.96, 'depth': 8},
    {'name': 'layer2.1.conv2', 'accuracy': 28.66, 'depth': 9},
    {'name': 'layer3.0.conv1', 'accuracy': 35.13, 'depth': 10},
    {'name': 'layer3.0.conv2', 'accuracy': 41.67, 'depth': 11},
    {'name': 'layer3.1.conv1', 'accuracy': 35.04, 'depth': 12},
    {'name': 'layer3.1.conv2', 'accuracy': 39.85, 'depth': 13},
    {'name': 'layer4.0.conv1', 'accuracy': 43.81, 'depth': 14},
    {'name': 'layer4.0.conv2', 'accuracy': 48.50, 'depth': 15},
    {'name': 'layer4.1.conv1', 'accuracy': 7.68,  'depth': 16},
    {'name': 'layer4.1.conv2', 'accuracy': 58.58, 'depth': 17}
]

# Normalize accuracies and depths
max_depth = max(layer['depth'] for layer in layer_data)
for layer in layer_data:
    layer['accuracy_norm'] = layer['accuracy'] / 100.0
    layer['depth_norm'] = (layer['depth'] - 1) / (max_depth - 1)
    # print(layer['depth_norm'])
    layer['benefit_score'] = layer['accuracy_norm'] * (1 - layer['depth_norm']*0.85)

# Filter out layers with very low accuracy (e.g., less than 20%)
min_accuracy_threshold = 20.0
filtered_layers = [layer for layer in layer_data if layer['accuracy'] >= min_accuracy_threshold]

# Rank layers by benefit score
ranked_layers = sorted(filtered_layers, key=lambda x: x['benefit_score'], reverse=True)

# Display the ranked layers
print("Recommended layers for branching (from most to least suitable):")
for layer in ranked_layers:
    print(f"{layer['name']} - Benefit Score: {layer['benefit_score']:.4f}, Accuracy: {layer['accuracy']}%, Depth: {layer['depth']}")
