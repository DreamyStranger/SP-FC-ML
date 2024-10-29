import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_score
import os

class DTClassifier:
    """
    A class to perform classification on accelerometer feature data using a Decision Tree.
    Outputs include accuracy, classification report, and confusion matrix.
    """

    def __init__(self, feature_dir, results_file):
        """
        Initializes the classifier with the feature dataset directory and results file path.

        Parameters:
        - feature_dir (str): Directory containing the CSV files for each time slice's basic and extended dataset.
        - results_file (str): Path to save the accuracy results for basic datasets.
        """
        self.feature_dir = feature_dir
        self.results_file = results_file
        self.extended_results_file = results_file.replace(".txt", "_ext.txt")  # Create extended results file name
        self.best_accuracy = 0
        self.best_file = None
        self.extended_best_accuracy = 0
        self.extended_best_file = None
        self.model = DecisionTreeClassifier(random_state=42)

        # Ensure results directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

    def load_data(self, feature_file):
        """
        Loads the feature dataset from a CSV file and separates it into features and labels.
        """
        data = pd.read_csv(feature_file)
        X = data.drop(columns=["Activity"])
        y = data["Activity"]
        return X, y

    def evaluate_file(self, feature_file, cv=10):
        """
        Loads data, trains, and evaluates the Decision Tree model for a specific feature file using cross-validation.

        Parameters:
        - feature_file (str): Path to the CSV file containing extracted features and labels.
        - cv (int): Number of cross-validation folds (default: 5).

        Returns:
        - mean_accuracy (float): Mean accuracy across all folds.
        - accuracy_scores (list): List of accuracy scores for each fold.
        """
        X, y = self.load_data(feature_file)

        # Use cross-validation to evaluate the model
        accuracy_scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        mean_accuracy = accuracy_scores.mean()

        # Generate a classification report based on the cross-validation (only for reference, not required here)
        # Note: cross_val_score only provides scores, not a report or confusion matrix
        report = f"Mean Accuracy: {mean_accuracy * 100:.2f}%\n" + \
                "Accuracy per Fold: " + ", ".join(f"{score * 100:.2f}%" for score in accuracy_scores)
        
        return mean_accuracy, report 

    def run_all_basic_datasets(self):
        """
        Runs the Decision Tree classifier on all basic feature datasets in the feature directory.
        Saves accuracy results to a file and identifies the best-performing dataset.
        """
        # Open the results file
        with open(self.results_file, "w") as results:
            results.write("Accuracy Results for Basic Datasets:\n")
            results.write("===================================\n\n")

            for file_name in os.listdir(self.feature_dir):
                if file_name.startswith("features_") and not file_name.startswith("features_ext"):
                    feature_file = os.path.join(self.feature_dir, file_name)

                    # Evaluate the current feature file
                    accuracy, report = self.evaluate_file(feature_file)

                    # Write results to file
                    results.write(f"File: {file_name}\n")
                    results.write(f"Accuracy: {accuracy * 100:.2f}%\n")
                    results.write("Classification Report:\n")
                    results.write(report)
                    results.write("\n" + "="*40 + "\n\n")

                    # Track the best model
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_file = file_name

            # Write the best file information to the results file
            results.write("Best Performing Basic Dataset:\n")
            results.write(f"File: {self.best_file}\n")
            results.write(f"Accuracy: {self.best_accuracy * 100:.2f}%\n")
            print(f"Best basic dataset file: {self.best_file} with accuracy: {self.best_accuracy * 100:.2f}%")

    def run_all_extended_datasets(self):
        """
        Runs the Decision Tree classifier on all extended feature datasets in the feature directory.
        Saves accuracy results to a separate file and compares with the best result from basic datasets.
        """
        # Open the extended results file
        with open(self.extended_results_file, "w") as results:
            results.write("Accuracy Results for Extended Datasets:\n")
            results.write("======================================\n\n")

            for file_name in os.listdir(self.feature_dir):
                if file_name.startswith("features_ext_"):
                    feature_file = os.path.join(self.feature_dir, file_name)

                    # Evaluate the current extended feature file
                    accuracy, report = self.evaluate_file(feature_file)

                    # Write results to file
                    results.write(f"File: {file_name}\n")
                    results.write(f"Accuracy: {accuracy * 100:.2f}%\n")
                    results.write("Classification Report:\n")
                    results.write(report)
                    results.write("\n" + "="*40 + "\n\n")

                    # Track the best extended model
                    if accuracy > self.extended_best_accuracy:
                        self.extended_best_accuracy = accuracy
                        self.extended_best_file = file_name

            # Write the best extended file information to the extended results file
            results.write("Best Performing Extended Dataset:\n")
            results.write(f"File: {self.extended_best_file}\n")
            results.write(f"Accuracy: {self.extended_best_accuracy * 100:.2f}%\n")
            print(f"Best extended dataset file: {self.extended_best_file} with accuracy: {self.extended_best_accuracy * 100:.2f}%")

    def save_best_datasets(self):
        """
        Saves the best-performing datasets for both basic and extended features for further processing.
        """
        if self.best_file:
            print(f"Best basic dataset selected: {self.best_file} with accuracy {self.best_accuracy * 100:.2f}%")
        else:
            print("No best basic dataset identified. Please run 'run_all_basic_datasets()' first.")
            
        if self.extended_best_file:
            print(f"Best extended dataset selected: {self.extended_best_file} with accuracy {self.extended_best_accuracy * 100:.2f}%")
        else:
            print("No best extended dataset identified. Please run 'run_all_extended_datasets()' first.")

    def perform_sfs(self):
        """
        Runs Sequential Feature Selection on the best-performing dataset overall.
        Uses cross-validation within SFS to select features and track performance at each step.
        """
        # Determine the file with the highest accuracy
        if self.best_accuracy >= self.extended_best_accuracy:
            best_file = os.path.join(self.feature_dir, self.best_file)
        else:
            best_file = os.path.join(self.feature_dir, self.extended_best_file) 
        
        print(f"Performing SFS on the best dataset: {best_file}")

        # Define result file for SFS
        sfs_extended_results = self.results_file.replace(".txt", "_sfs.txt")

        # Load data for the file
        X, y = self.load_data(best_file)
        
        # Set up cross-validation with a fixed random seed
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        # Initialize Sequential Feature Selector with cross-validation
        sfs = SequentialFeatureSelector(
            self.model,
            n_features_to_select="auto",  # Let SFS choose the best subset automatically
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
            scores = cross_val_score(self.model, X[temp_features], y, cv=cv, scoring="accuracy")
            step_accuracies.append(scores.mean())

        # Save detailed SFS results to the specified result file
        with open(sfs_extended_results, "w") as results:
            results.write("Sequential Feature Selection Results:\n")
            results.write("=========================================================\n\n")
            results.write(f"File: {best_file}\n")
            results.write(f"Final Selected Features: {selected_features}\n")
            results.write(f"Final Accuracy with Selected Features: {step_accuracies[-1] * 100:.2f}%\n")
            
            # Log accuracy at each step as features are added
            results.write("\nAccuracy at Each Step of SFS:\n")
            for step, (feature, accuracy) in enumerate(zip(selected_features, step_accuracies), 1):
                features_subset.append(feature)
                results.write(f"Step {step}: Selected Feature = {features_subset}, Accuracy = {accuracy * 100:.2f}%\n")
            
            results.write("\n" + "="*40 + "\n\n")

        # Print final results
        print(f"Final selected features from {best_file}: {selected_features}")
        print(f"Final cross-validation accuracy with selected features from {best_file}: {step_accuracies[-1] * 100:.2f}%")
        print("Detailed SFS process with cross-validation has been saved to the results file.")


    def run(self):
        """
        Runs the complete classification and evaluation process:
        1. Evaluates all basic datasets and identifies the best-performing basic dataset.
        2. Evaluates all extended datasets and identifies the best-performing extended dataset.
        3. Displays the best-performing datasets for both basic and extended features.
        4. Runs Sequential Feature Selection (SFS) on the best extended dataset.
        """
        print("Starting evaluation for basic datasets...")
        self.run_all_basic_datasets()  # Evaluate and save results for basic datasets
        print("Basic datasets evaluation complete.\n")

        print("Starting evaluation for extended datasets...")
        self.run_all_extended_datasets()  # Evaluate and save results for extended datasets
        print("Extended datasets evaluation complete.\n")

        print("Best-performing datasets summary:")
        self.save_best_datasets()  # Display and save the best basic and extended datasets

        # Run SFS on the best dataset overall
        print("\nStarting Sequential Feature Selection (SFS) on the best dataset...")
        self.perform_sfs() 
        print("Sequential Feature Selection complete.\n")


def main():
    # Define the directory containing the feature CSV files
    feature_dir = "data/features"  # Adjust this path based on where your feature files are stored
    
    # Define the output file name for storing results
    results_file = "results/accuracy_results.txt"  # This will store results for basic datasets, with a separate file for extended
    
    # Create an instance of DTClassifier
    classifier = DTClassifier(feature_dir, results_file)
    
    # Run the complete evaluation process
    classifier.run()

if __name__ == "__main__":
    main()