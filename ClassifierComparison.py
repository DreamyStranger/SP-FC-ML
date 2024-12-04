import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
import os
import json


class ClassifierSFS:
    def __init__(self, feature_dir, results_dir):
        """
        Initializes the classifier comparison for SFS with multiple models.

        Parameters:
        - feature_dir (str): Directory containing the CSV files for datasets.
        - results_dir (str): Directory to save SFS results.
        """
        self.feature_dir = feature_dir
        self.results_dir = results_dir
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(kernel="linear", random_state=42) 
        }
        self.datasets = self.get_all_datasets()
        self.results = {}  # In-memory storage for results

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def get_all_datasets(self):
        """Lists all CSV files in the feature directory."""
        datasets = []
        for file_name in os.listdir(self.feature_dir):
            if file_name.endswith(".csv"):
                dataset_path = os.path.join(self.feature_dir, file_name)
                datasets.append((dataset_path, file_name))  # Store path and file name
        return datasets

    def load_data(self, feature_file):
        """Loads the dataset from a CSV file and separates features and labels."""
        data = pd.read_csv(feature_file)
        X = data.drop(columns=["label"])
        y = data["label"]
        return X, y

    def perform_sfs(self, model, X, y, cv):
        """
        Performs Sequential Feature Selection (SFS) for a specific model and dataset.
        """
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select="auto",
            direction="forward",
            scoring="accuracy",
            cv=cv
        )

        # Fit SFS and extract selected features
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()].tolist()

        # Compute accuracy as features are added
        step_accuracies = []
        for i in range(1, len(selected_features) + 1):
            temp_features = selected_features[:i]
            scores = cross_val_score(model, X[temp_features], y, cv=cv, scoring="accuracy")
            step_accuracies.append(scores.mean())

        return selected_features, step_accuracies

    def run(self):
        """Runs SFS for all models and datasets and saves results to JSON."""
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        for file_path, dataset_label in self.datasets:
            print(f"Processing {dataset_label}: {file_path}")
            X, y = self.load_data(file_path)

            self.results[dataset_label] = {}
            for model_name, model in self.models.items():
                print(f"Running SFS with {model_name} on {dataset_label}...")
                selected_features, step_accuracies = self.perform_sfs(model, X, y, cv)

                # Store results in memory
                self.results[dataset_label][model_name] = {
                    "selected_features": selected_features,
                    "step_accuracies": [acc * 100 for acc in step_accuracies]  # Save as percentages
                }

                print(f"{model_name} - {dataset_label}: Final accuracy = {step_accuracies[-1] * 100:.2f}%")

        # Save results to JSON file
        json_file = os.path.join(self.results_dir, "sfs_results.json")
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"SFS results saved to {json_file}")

def main():
    """
    Main function to execute Sequential Feature Selection (SFS) analysis.
    """
    # Define the directory containing dataset CSV files
    feature_dir = "data/features" 

    # Define the directory to save results and JSON file
    results_dir = "results"  # Directory to store JSON results

    # Create an instance of ClassifierSFS
    sfs_comparison = ClassifierSFS(feature_dir, results_dir)

    # Run the SFS analysis
    sfs_comparison.run()

if __name__ == "__main__":
    main()
