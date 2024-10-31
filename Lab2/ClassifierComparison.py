import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, KFold
import os
from sklearn.tree import DecisionTreeClassifier

class ClassifierComparison:
    def __init__(self, feature_dir, results_file):
        """
        Initializes the classifier comparison for SFS with multiple models.

        Parameters:
        - feature_dir (str): Directory containing the CSV files for 4s and 3s extended datasets.
        - results_file (str): Path to save the accuracy results for each classifier.
        """
        self.feature_dir = feature_dir
        self.results_file = results_file
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(kernel="linear", random_state=42)  # Linear kernel for simplicity in feature selection
        }
        self.files_to_process = [
            (os.path.join(self.feature_dir, "features_ext_4s.csv"), "4-second"),
            (os.path.join(self.feature_dir, "features_ext_3s.csv"), "3-second")
        ]

        # Ensure results directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

    def load_data(self, feature_file):
        """Loads the feature dataset from a CSV file and separates it into features and labels."""
        data = pd.read_csv(feature_file)
        X = data.drop(columns=["Activity"])
        y = data["Activity"]
        return X, y

    def perform_sfs_for_model(self, model, X, y, cv, model_name, file_label, results):
        """Performs SFS for a given model and dataset, and writes results to the output file."""
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select="auto",
            direction="forward",
            scoring="accuracy",
            cv=cv
        )
        
        # Fit the SFS model
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()].tolist()
        
        # Track accuracies as features are added
        step_accuracies = []
        features_subset = []

        for i in range(1, len(selected_features) + 1):
            temp_features = selected_features[:i]
            scores = cross_val_score(model, X[temp_features], y, cv=cv, scoring="accuracy")
            step_accuracies.append(scores.mean())

        # Write results to file
        results.write(f"\n{model_name} Results for {file_label} dataset:\n")
        results.write("=========================================================\n")
        results.write(f"Final Selected Features: {selected_features}\n")
        results.write(f"Final Accuracy with Selected Features: {step_accuracies[-1] * 100:.2f}%\n")
        
        results.write("\nAccuracy at Each Step of SFS:\n")
        for step, (feature, accuracy) in enumerate(zip(selected_features, step_accuracies), 1):
            features_subset.append(feature)
            results.write(f"Step {step}: Selected Feature = {features_subset}, Accuracy = {accuracy * 100:.2f}%\n")
        results.write("\n" + "="*40 + "\n\n")

        # Print final results for this dataset
        print(f"{model_name} - {file_label} dataset: Final selected features: {selected_features}")
        print(f"{model_name} - {file_label} dataset: Final accuracy: {step_accuracies[-1] * 100:.2f}%")

    def run(self):
        """Runs SFS on each model and dataset, and writes the results to the same file."""
        with open(self.results_file, "w") as results:
            results.write("Sequential Feature Selection (SFS) Results for Different Models:\n")
            results.write("===============================================================\n\n")

            # Set up cross-validation
            cv = KFold(n_splits=10, shuffle=True, random_state=42)

            for file_path, file_label in self.files_to_process:
                print(f"\nProcessing {file_label} dataset: {file_path}")
                X, y = self.load_data(file_path)

                for model_name, model in self.models.items():
                    print(f"Performing SFS with {model_name} on {file_label} dataset...")
                    self.perform_sfs_for_model(model, X, y, cv, model_name, file_label, results)

            # Summarize the best classifier based on final accuracy
            print("SFS with all classifiers complete. Check results for performance comparison.")

# Define the directory containing the feature CSV files
feature_dir = "data/features"  # Adjust path based on your feature files location

# Define the output file name for storing results
results_file = "results/sfs_comparison_results.txt"

# Create an instance of ClassifierComparison and run the comparison
comparison = ClassifierComparison(feature_dir, results_file)
comparison.run()
