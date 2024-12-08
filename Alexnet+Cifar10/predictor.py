# Define class names
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Class-wise accuracies for the main model
main_accuracies = {
    0: 76.50,  # airplane
    1: 79.30,  # automobile
    2: 46.20,  # bird
    3: 53.60,  # cat
    4: 56.20,  # deer
    5: 55.20,  # dog
    6: 75.50,  # frog
    7: 69.70,  # horse
    8: 73.90,  # ship
    9: 74.50   # truck
}

# Class-wise accuracies for each convolutional layer
layer_accuracies = {
    'conv1': {
        0: 62.20, 1: 73.00, 2: 42.20, 3: 29.60, 4: 44.70,
        5: 48.80, 6: 50.80, 7: 58.70, 8: 66.20, 9: 49.00
    },
    'conv2': {
        0: 39.00, 1: 52.90, 2: 39.20, 3: 29.90, 4: 37.30,
        5: 32.90, 6: 51.60, 7: 63.40, 8: 72.80, 9: 60.10
    },
    'conv3': {
        0: 60.90, 1: 66.40, 2: 40.70, 3: 43.60, 4: 52.10,
        5: 46.00, 6: 59.40, 7: 59.50, 8: 68.80, 9: 68.70
    },
    'conv4': {
        0: 60.80, 1: 62.10, 2: 41.60, 3: 33.40, 4: 58.60,
        5: 40.90, 6: 68.00, 7: 61.40, 8: 76.90, 9: 69.40
    },
    'conv5': {
        0: 63.70, 1: 66.40, 2: 47.30, 3: 39.30, 4: 53.60,
        5: 47.80, 6: 64.20, 7: 62.70, 8: 73.00, 9: 64.00
    }
}

# List of layers
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# Initialize a dictionary to store benefit scores
benefit_scores = {}

# Calculate benefit scores for each class at each layer
for layer in layers:
    benefit_scores[layer] = {}
    for class_id in range(len(classes)):
        layer_acc = layer_accuracies[layer][class_id]
        main_acc = main_accuracies[class_id]
        # Benefit Score: Ratio of layer accuracy to main model accuracy
        benefit_score = layer_acc / main_acc
        benefit_scores[layer][class_id] = benefit_score

# Function to display benefit scores for each class
def display_benefit_scores_per_class():
    for class_id in range(len(classes)):
        class_name = classes[class_id]
        print(f"Class {class_id} ({class_name}):")
        # Collect benefit scores for this class across layers
        class_benefit_scores = []
        for layer in layers:
            benefit_score = benefit_scores[layer][class_id]
            class_acc = layer_accuracies[layer][class_id]
            class_benefit_scores.append((layer, benefit_score, class_acc))
        # Sort layers by benefit score descending
        class_benefit_scores.sort(key=lambda x: x[1], reverse=True)
        # Print the layers and benefit scores
        for layer_name, score, acc in class_benefit_scores:
            print(f"  Layer {layer_name}: Benefit Score = {score:.2f}, "
                  f"Layer Accuracy = {acc}%, Main Accuracy = {main_accuracies[class_id]}%")
        print()

# Function to display average benefit scores per layer
def display_average_benefit_scores():
    average_benefit_scores = {}
    for layer in layers:
        total_score = sum(benefit_scores[layer].values())
        average_score = total_score / len(classes)
        average_benefit_scores[layer] = average_score

    # Sort layers by average benefit score
    sorted_layers = sorted(average_benefit_scores.items(), key=lambda x: x[1], reverse=True)
    print("Average Benefit Scores per Layer:")
    for layer_name, avg_score in sorted_layers:
        print(f"  Layer {layer_name}: Average Benefit Score = {avg_score:.2f}")
    print()

# Display the benefit scores
display_benefit_scores_per_class()
display_average_benefit_scores()
