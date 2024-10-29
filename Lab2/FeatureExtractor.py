import pandas as pd
import os
import glob
import numpy as np

class AccelerometerFeatureExtractor:
    """
    A class to extract features from multiple accelerometer data files in specified directories.
    Extracted features include mean, standard deviation, median, and root mean square (RMS) of each axis (X, Y, Z).
    """

    def __init__(self, hand_wash_dir, non_hand_wash_dir, sample_rate=100):
        """
        Initializes the feature extractor with directories containing multiple files
        and sample rate.

        Parameters:
        - hand_wash_dir (str): Directory path containing hand-wash accelerometer data files.
        - non_hand_wash_dir (str): Directory path containing non-hand-wash accelerometer data files.
        - sample_rate (int): Number of samples per second (default: 100, assuming 100 Hz sample rate).
        """
        self.hand_wash_dir = hand_wash_dir
        self.non_hand_wash_dir = non_hand_wash_dir
        self.sample_rate = sample_rate
        self.features_df = None

    def load_data(self, directory):
        """
        Loads all CSV files from a specified directory into a single DataFrame.

        Parameters:
        - directory (str): Path to the directory containing CSV files.

        Returns:
        - DataFrame: Combined DataFrame of all files in the directory.

        Raises:
        - ValueError: If no CSV files are found in the directory.
        """
        # Find all CSV files in the specified directory
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        if not all_files:
            raise ValueError(f"No CSV files found in directory: {directory}")

        # Read and concatenate all CSV files into a single DataFrame
        data_list = [pd.read_csv(file) for file in all_files]
        combined_data = pd.concat(data_list, ignore_index=True)

        #print(f"Combined data shape: {combined_data.shape}")  # Debugging line to verify combined data
        return combined_data


    def extract_features(self, data, label, time_slice=1):
        """
        Extracts mean, standard deviation, median, and root mean square (RMS) features for each time slice window in the accelerometer data,
        with a sliding window of 1 second.

        Parameters:
        - data (DataFrame): DataFrame containing accelerometer data with X, Y, Z columns.
        - label (str): Label for the activity type ('hand_wash' or 'no_hand_wash').
        - time_slice (int): Duration of the time slice in seconds (1, 2, 3, or 4).

        Returns:
        - features (list): List of feature vectors for each window, each containing
        mean, std, median, rms for X, Y, Z axes, and the label.
        """
        # Assume the last three columns are X, Y, Z accelerometer data
        columns = data.columns[-3:]
        features = []

        # Calculate window size in terms of samples for the given time slice
        window_size = self.sample_rate * time_slice
        step_size = self.sample_rate  # Sliding by 1-second increments

        # Iterate through data in windows with a 1-second sliding window
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size][columns]
            
            # Skip processing if the window is empty
            if window.empty:
                continue
            
            # Compute mean, standard deviation, median, and RMS for each axis
            mean_x, std_x = window[columns[0]].mean(), window[columns[0]].std()
            median_x = window[columns[0]].median()
            rms_x = np.sqrt(np.mean(window[columns[0]] ** 2))

            mean_y, std_y = window[columns[1]].mean(), window[columns[1]].std()
            median_y = window[columns[1]].median()
            rms_y = np.sqrt(np.mean(window[columns[1]] ** 2))

            mean_z, std_z = window[columns[2]].mean(), window[columns[2]].std()
            median_z = window[columns[2]].median()
            rms_z = np.sqrt(np.mean(window[columns[2]] ** 2))
            
            # Append features and label to the feature list
            features.append([
                mean_x, std_x, median_x, rms_x,
                mean_y, std_y, median_y, rms_y,
                mean_z, std_z, median_z, rms_z,
                label
            ])

        return features


    def process_data(self, time_slice=1):
        """
        Processes data from multiple files in hand-wash and non-hand-wash directories by extracting features.
        Combines all extracted features into a single DataFrame with labeled activities.

        Parameters:
        - time_slice (int): Duration of the time slice in seconds (1, 2, 3, or 4).
        """
        # Load data from each directory
        hand_wash_data = self.load_data(self.hand_wash_dir)
        non_hand_wash_data = self.load_data(self.non_hand_wash_dir)
        
        # Extract features for hand-wash and non-hand-wash datasets
        hand_wash_features = self.extract_features(hand_wash_data, "hand_wash", time_slice)
        non_hand_wash_features = self.extract_features(non_hand_wash_data, "no_hand_wash", time_slice)
        
        # Combine features into a DataFrame
        columns = [
            "mean_x", "std_x", "median_x", "rms_x",
            "mean_y", "std_y", "median_y", "rms_y",
            "mean_z", "std_z", "median_z", "rms_z",
            "Activity"
        ]
        self.features_df = pd.DataFrame(hand_wash_features + non_hand_wash_features, columns=columns)

    def save_features(self, output_dir, time_slice):
        """
        Saves both the basic and extended feature sets to the specified output directory.

        Parameters:
        - output_dir (str): Directory path where feature files will be saved.
        - time_slice (int): Duration of the time slice in seconds (1, 2, 3, or 4).
        """
        if self.features_df is not None:
            os.makedirs(output_dir, exist_ok=True)

            # Save basic feature set (mean, std, Activity)
            basic_df = self.features_df[[
                "mean_x", "std_x",
                "mean_y", "std_y",
                "mean_z", "std_z",
                "Activity"
            ]]
            basic_output_file = os.path.join(output_dir, f"features_{time_slice}s.csv")
            basic_df.to_csv(basic_output_file, index=False)
            print(f"Basic features saved to {basic_output_file}")

            # Save extended feature set (mean, std, median, rms, Activity)
            extended_output_file = os.path.join(output_dir, f"features_ext_{time_slice}s.csv")
            self.features_df.to_csv(extended_output_file, index=False)
            print(f"Extended features saved to {extended_output_file}")
        else:
            print("Feature extraction not completed. Run process_data() first.")

    def run(self, output_dir="data/features", time_slice=1):
        """
        Executes the complete feature extraction process, from loading data to saving features.

        Parameters:
        - output_dir (str): Directory path where feature files will be saved.
        - time_slice (int): Duration of the time slice in seconds (1, 2, 3, or 4).
        """
        self.process_data(time_slice)
        self.save_features(output_dir, time_slice)


def main():
    # Define directories for input data and output features
    hand_wash_dir = "data/raw/hand_wash"
    non_hand_wash_dir = "data/raw/not_hand_wash"
    output_dir = "data/features"

    # Create an instance of AccelerometerFeatureExtractor
    feature_extractor = AccelerometerFeatureExtractor(hand_wash_dir, non_hand_wash_dir)

    # Define time slices to process
    time_slices = [1, 2, 3, 4]

    # Run the feature extraction for each time slice
    for time_slice in time_slices:
        feature_extractor.run(output_dir=output_dir, time_slice=time_slice)

if __name__ == "__main__":
    main()