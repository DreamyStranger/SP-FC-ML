import json
import os
import matplotlib.pyplot as plt


def plot_sfs_results_from_json(json_file, output_file="sfs_combined_plot.png"):
    # Load JSON results
    with open(json_file, "r") as f:
        results = json.load(f)
    
    # Extract unique models and datasets
    datasets = list(results.keys())
    models = list(next(iter(results.values())).keys())
    
    # Create the overall figure with a wide row layout for 3 model-specific sections
    fig = plt.figure(figsize=(22, 8))  # Keep the overall figure size the same
    outer_gs = plt.GridSpec(1, 3, figure=fig, wspace=0.2)  # Reduce the space between sections

    for model_idx, model in enumerate(models):
        # Create an outer subplot for each model
        inner_gs = outer_gs[model_idx].subgridspec(2, 2, wspace=0.3, hspace=0.4)  # Adjust grid spacing

        for dataset_idx, dataset in enumerate(datasets):
            if dataset_idx >= 4:  # Limit to 4 datasets per model
                break

            # Determine the row and column for this dataset
            row, col = divmod(dataset_idx, 2)
            ax = fig.add_subplot(inner_gs[row, col])

            # Extract step accuracies and features for the model-dataset combination
            step_accuracies = results[dataset][model]["step_accuracies"]
            selected_features = results[dataset][model]["selected_features"]

            # Plot step accuracies
            x_vals = range(1, len(step_accuracies) + 1)
            y_vals = step_accuracies
            ax.plot(x_vals, y_vals, marker="o", color="blue")
            ax.set_title(f"{dataset}", fontsize=10)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(selected_features, rotation=45, fontsize=6)
            ax.grid(True)
            ax.set_ylim(83, 100)  # Assuming accuracy percentages

            # Add accuracy values on the points
            for x, y in zip(x_vals, y_vals):
                ax.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom", color="black")

        # Add a title for the model section
        fig.text(0.33 * model_idx + 0.17, 0.92, f"{model}", ha="center", fontsize=10, fontweight="bold")

    # Adjust layout and save the figure
    fig.suptitle("Sequential Feature Selection Comparison Across Models and Datasets", fontsize=12)
    plt.tight_layout()  # Adjust the layout to reduce whitespace
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Combined plot saved to: {output_file}")


def main():
    """
    Main function to parse JSON results and generate SFS comparison plots.
    """
    # Define paths
    results_dir = "results"  # Directory where the JSON and plot files are saved
    json_file = os.path.join(results_dir, "sfs_results.json")  # Path to the JSON results file
    plot_file = os.path.join(results_dir, "sfs_combined_plot.png")  # Path to save the combined plot

    # Check if the results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist.")
        return

    # Check if the JSON file exists
    if not os.path.isfile(json_file):
        print(f"JSON file '{json_file}' does not exist.")
        return

    # Generate the combined plot
    print(f"Generating combined plot from JSON file: {json_file}")
    plot_sfs_results_from_json(json_file, output_file=plot_file)
    print(f"Combined plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
