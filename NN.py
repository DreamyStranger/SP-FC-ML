import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(dim=-1)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs).squeeze()
            predictions = (outputs > 0.5).int()
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.tolist())
    return y_true, y_pred

def get_best_dataset_and_features(json_file):
    with open(json_file, "r") as f:
        sfs_results = json.load(f)

    best_dataset = None
    best_model = None
    best_accuracy = 0
    selected_features = []

    for dataset, models in sfs_results.items():
        for model, metrics in models.items():
            final_accuracy = metrics["step_accuracies"][-1]
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_dataset = dataset
                best_model = model
                selected_features = metrics["selected_features"]

    return best_dataset, selected_features

def hyperparameter_tuning(X_train, X_test, y_train, y_test, hidden_dim_options, lr_options, epochs):
    results = []

    for hidden_dim1, hidden_dim2 in hidden_dim_options:
        for lr in lr_options:
            # Create DataLoader
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize model, criterion, and optimizer
            input_dim = X_train.shape[1]
            model = NeuralNet(input_dim, hidden_dim1, hidden_dim2)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train and evaluate the model
            model = train_model(model, train_loader, criterion, optimizer, epochs)
            y_true, y_pred = evaluate_model(model, test_loader)

            # Calculate accuracy
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            results.append({
                "hidden_dim1": hidden_dim1,
                "hidden_dim2": hidden_dim2,
                "learning_rate": lr,
                "accuracy": accuracy
            })

    return results

def plot_hyperparameter_results(results, output_file="hyperparameter_tuning_plot.png"):
    # Prepare data for plotting
    configurations = [f"H1={r['hidden_dim1']}, H2={r['hidden_dim2']}, LR={r['learning_rate']:.4f}" for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]

    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(configurations, accuracies, color="skyblue")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Accuracy (%)")
    plt.title("Hyperparameter Tuning Results")

    # Add accuracy labels to bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{accuracy:.2f}%", ha="center", va="bottom", fontsize=8)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def main():
    # Paths
    sfs_json_file = "results/sfs_results.json"
    feature_dir = "data/features"
    output_file = "NN_hyperparameter_results.txt"
    plot_file = "results/hyperparameter_tuning_plot.png"

    # Get best dataset and features
    best_dataset, selected_features = get_best_dataset_and_features(sfs_json_file)
    dataset_path = f"{feature_dir}/{best_dataset}"

    # Load and preprocess data
    data = pd.read_csv(dataset_path)
    X = data[selected_features]
    y = LabelEncoder().fit_transform(data["label"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define hyperparameter options
    hidden_dim_options = [(64, 32), (128, 64), (256, 128)]
    lr_options = [0.01, 0.001, 0.0001]
    epochs = 10

    # Perform hyperparameter tuning
    results = hyperparameter_tuning(X_train, X_test, y_train, y_test, hidden_dim_options, lr_options, epochs)

    # Save results to a file
    with open(output_file, "w") as f:
        for r in results:
            f.write(f"H1={r['hidden_dim1']}, H2={r['hidden_dim2']}, LR={r['learning_rate']:.4f}, Accuracy={r['accuracy']:.5f}\n")

    # Plot the results
    plot_hyperparameter_results(results, output_file=plot_file)

    print(f"Results saved to {output_file}")
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    main()
