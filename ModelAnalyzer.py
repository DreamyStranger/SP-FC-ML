import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ModelAnalyzer:
    def __init__(self, feature_dir, model):
        self.feature_dir = feature_dir
        self.model = model  # Dynamically assign the model
        self.best_accuracy = 0
        self.best_file = None
        self.results_dir = f"results/{type(model).__name__}"  # Directory based on model name
        os.makedirs(self.results_dir, exist_ok=True)  # Ensure the directory exists

    def load_data(self, feature_file):
        data = pd.read_csv(feature_file)
        X = data.drop(columns=["label"])
        y = data["label"]
        return X, y

    def evaluate_file(self, feature_file, cv=10):
        X, y = self.load_data(feature_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)
        accuracy_scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        mean_accuracy = accuracy_scores.mean()
        return mean_accuracy, report, confusion

    def plot_bar_chart(self, dataset_names, mean_accuracies):
        plt.figure(figsize=(8, 4))
        bars = plt.barh(dataset_names, mean_accuracies, color="skyblue")
        plt.xlabel("Mean Accuracy")
        plt.ylabel("Datasets")
        plt.title("Mean Accuracy Across Datasets")
        plt.xlim(0, 1)

        for bar, accuracy in zip(bars, mean_accuracies):
            plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{accuracy * 100:.2f}%", va='center', ha='right', color="black", fontsize=10, weight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/horizontal_bar_chart.png")
        plt.show()

    def plot_box_plot(self, flattened_data):
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=flattened_data["Dataset"], y=flattened_data["Accuracy"], hue=flattened_data["Dataset"], palette="Set2", dodge=False)
        plt.xlabel("Datasets")
        plt.ylabel("Cross-Validation Accuracy")
        plt.title("Cross-Validation Accuracy Distribution")
        plt.xticks(rotation=45)
        plt.legend([], [], frameon=False)  # Remove legend if not needed
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/box_plot.png")
        plt.show()


    def plot_combined_heatmaps(self, results):
        num_datasets = len(results)
        cols = 1
        rows = num_datasets

        fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * num_datasets))
        if num_datasets == 1:
            axes = [axes]

        for idx, (dataset_name, (_, _, confusion)) in enumerate(results.items()):
            sns.heatmap(confusion, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[idx],
                        xticklabels=["Non-Active", "Active"],
                        yticklabels=["Non-Active", "Active"])
            axes[idx].set_title(dataset_name, fontsize=12)
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")

        fig.suptitle("Confusion Matrix Heatmaps", y=0.99)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/combined_confusion_matrices.png")
        plt.show()

    def plot_combined_metrics_heatmap(self, results):
        metrics = ["Precision", "Recall", "F1-Score"]
        combined_metrics = []

        for dataset_name, (_, report, _) in results.items():
            dataset_metrics = [
                report[label][metric.lower()] for label in report if label not in {"accuracy", "macro avg", "weighted avg"} 
                for metric in metrics
            ]
            combined_metrics.append(dataset_metrics)

        combined_df = pd.DataFrame(combined_metrics,
                                   index=results.keys(),
                                   columns=[f"{metric} ({label})" for label in report if label not in {"accuracy", "macro avg", "weighted avg"} for metric in metrics])

        plt.figure(figsize=(5, 8))
        sns.heatmap(combined_df, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Score'})
        plt.title("Metrics Heatmap (Precision, Recall, F1-Score)")
        plt.xlabel("Metrics")
        plt.ylabel("Datasets")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/combined_metrics_heatmap.png")
        plt.show()

    def visualize(self, results):
        dataset_names = list(results.keys())
        mean_accuracies = [sum(scores) / len(scores) for scores, _, _ in results.values()]
        self.plot_bar_chart(dataset_names, mean_accuracies)

        flattened_data = {
            "Dataset": [name for name, (scores, _, _) in results.items() for _ in scores],
            "Accuracy": [score for scores, _, _ in results.values() for score in scores]
        }
        self.plot_box_plot(flattened_data)
        self.plot_combined_heatmaps(results)
        self.plot_combined_metrics_heatmap(results)

    def run(self):
        results = {}

        for file_name in os.listdir(self.feature_dir):
            if file_name.endswith(".csv"):
                feature_file = os.path.join(self.feature_dir, file_name)
                accuracy, report, confusion = self.evaluate_file(feature_file)
                results[file_name] = (cross_val_score(self.model, *self.load_data(feature_file), cv=10, scoring="accuracy").tolist(),
                                      report, confusion)

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_file = file_name

        print(f"Best dataset file: {self.best_file} with accuracy: {self.best_accuracy * 100:.2f}%")
        self.visualize(results)


def main():
    feature_dir = "data/features"

    print("Select a model for analysis:")
    print("0: Decision Tree")
    print("1: Random Forest")
    print("2: Support Vector Machine")

    try:
        choice = int(input("Enter your choice (0, 1, or 2): "))
        if choice == 0:
            model = DecisionTreeClassifier(random_state=42)
            print("You selected: Decision Tree")
        elif choice == 1:
            model = RandomForestClassifier(random_state=42)
            print("You selected: Random Forest")
        elif choice == 2:
            model = SVC(random_state=42)
            print("You selected: Support Vector Machine")
        else:
            print("Invalid choice. Please run the program again.")
            return
    except ValueError:
        print("Invalid input. Please enter a number (0, 1, or 2).")
        return

    analyzer = ModelAnalyzer(feature_dir, model)
    analyzer.run()



if __name__ == "__main__":
    main()
