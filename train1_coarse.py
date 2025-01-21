import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def initialize_model(input_dim):
    """Initialize model parameters."""
    return {
        "weights": np.zeros(input_dim),
        "bias": 0
    }

def train(X, y, model, lr=0.01):
    """
    Perform a single epoch of training for logistic regression.
    
    Parameters:
        X (np.array): Input features (m x n).
        y (np.array): Binary labels (m,).
        model (dict): Model parameters containing 'weights' and 'bias'.
        lr (float): Learning rate.
    
    Returns:
        dict: Updated model parameters.
    """
    m = X.shape[0]
    # Linear prediction
    z = np.dot(X, model["weights"]) + model["bias"]
    # Activation
    predictions = sigmoid(z)
    # Compute gradients
    error = predictions - y
    dw = np.dot(X.T, error) / m
    db = np.sum(error) / m
    # Update weights and bias
    model["weights"] -= lr * dw
    model["bias"] -= lr * db
    return model

def predict(X, model):
    """Predict binary labels for the given input features."""
    z = np.dot(X, model["weights"]) + model["bias"]
    probs = sigmoid(z)
    ent = -np.sum(probs * np.log(probs))
    return (probs >= 0.5).astype(int), ent

def load_data(file_path):
    import pandas as pd
    data = pd.read_json(file_path, lines=True)
    X = np.array(data["embedding:3-large"].tolist())
    y = np.array(data["techniques"].tolist(), dtype=object)
    return X, y

import numpy as np

def oversample_data(X, y):
    """
    Oversample the dataset to balance the classes.
    
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Labels of shape (n_samples,).
    
    Returns:
        numpy.ndarray, numpy.ndarray: Balanced feature matrix and labels.
    """
    # Find unique classes and their counts
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)  # Maximum class count for balancing

    # Create arrays to store oversampled data
    X_oversampled = []
    y_oversampled = []

    # Oversample each class
    for class_label, count in zip(unique_classes, counts):
        # Filter samples for the current class
        X_class = X[y == class_label]
        y_class = y[y == class_label]

        # Calculate the number of times to replicate the class
        num_repeats = max_count // count
        remainder = max_count % count

        # Replicate the class samples
        X_class_repeated = np.tile(X_class, (num_repeats, 1))
        y_class_repeated = np.tile(y_class, num_repeats)

        # Add random samples for the remainder
        if remainder > 0:
            random_indices = np.random.choice(count, size=remainder, replace=False)
            X_class_remainder = X_class[random_indices]
            y_class_remainder = y_class[random_indices]

            # Combine the repeated and remainder samples
            X_class_repeated = np.vstack((X_class_repeated, X_class_remainder))
            y_class_repeated = np.hstack((y_class_repeated, y_class_remainder))

        # Append oversampled class data
        X_oversampled.append(X_class_repeated)
        y_oversampled.append(y_class_repeated)

    # Combine all classes
    X_balanced = np.vstack(X_oversampled)
    y_balanced = np.hstack(y_oversampled)

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffled_indices]
    y_balanced = y_balanced[shuffled_indices]

    return X_balanced, y_balanced

def prepare_data(X, y, label, seed=42):
    y_bin = np.array([1 if labels else 0 for labels in y])
    return train_test_split(X, y_bin, test_size=0.2, random_state=seed)

def run_logistic_regression(X, y, label, steps=1000, lr=1, batch_size=0, seed=42):
    X_train, X_test, y_train, y_test = prepare_data(X, y, label, seed=seed)
    print(label, 'test', np.unique(y_test, return_counts=True))
    print(label, 'train', np.unique(y_train, return_counts=True))
    X_train, y_train = oversample_data(X_train, y_train)
    print(label, 'oversampled train', np.unique(y_train, return_counts=True))
    if not batch_size:
        batch_size = 1.0
    if isinstance(batch_size, float):
        batch_size = int(len(X_train) * batch_size)
    print(f"Training for {label=} {steps=} {lr=} {batch_size=}")

    model = initialize_model(input_dim=X_train.shape[1])

    for step in range(steps):
        model = train(X_train, y_train, model, lr)

    predictions, entropy = predict(X_test, model)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=True)
    print(f"Results for {label} {entropy=:.2f}:")
    print(classification_report(y_test, predictions, zero_division=True))
    return report

if __name__ == "__main__":
    file_path = "data/train.emb.jsonl"
    X, y = load_data(file_path)
    results = []
    
    labels_list = [
        "sketchy",
    ]
    
    for lr in [10, 1, 0.1, 0.01, 0.001]:
        for batch_size in [16,32,64,128,256,512,1024]:
            for label in labels_list:
                results.append(dict(lr=lr, batch_size=batch_size, label=label, result=run_logistic_regression(X, y, label, lr=lr, batch_size=batch_size)))

    with open("exp/cv_logistic_coarse.json", "w") as f:
        print(json.dumps(results, ensure_ascii=False), file=f)
