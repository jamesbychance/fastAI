"""
This code creates a system for recognizing different exercises from your Garmin watch's motion data. 
It works by first reading the raw sensor data (accelerometer measurements) from your Garmin's FIT files and breaking it into small time windows - imagine taking a 2-second video clip of each exercise movement. 
For each window, it creates a special type of image called a spectrogram that shows the pattern of movement (like how a music visualizer shows patterns in sound). 
These images then go into a pre-trained neural network (using FastAI and the same approach as lecture 2) that learns to recognize the patterns specific to each exercise - for example, the distinct up-down pattern of a deadlift versus the pushing pattern of a kettlebell press. 
As a bonus feature, it also includes basic rep counting by looking for peaks in the movement pattern, like counting each time there's a significant up-down motion for deadlifts. 
It's structured as two main classes: one that handles processing the raw Garmin data into images (GarminDataProcessor), and another that handles the actual exercise recognition and rep counting (GarminHAR).
"""


from fastai.vision.all import *
import pandas as pd
import numpy as np
from scipy import signal
from fitparse import FitFile
import matplotlib.pyplot as plt
from pathlib import Path

# Config
WINDOW_SIZE = 50  # data points per window (adjust based on Garmin sampling rate)
OVERLAP = 25      # overlap between windows
ACTIVITIES = ['deadlift', 'kb_press', 'rest']  # extend as needed

class GarminDataProcessor:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.processed_path = self.data_path/'processed'
        self.processed_path.mkdir(exist_ok=True)
        
    def load_fit_file(self, fit_file):
        """Load and parse a single FIT file"""
        fitfile = FitFile(str(fit_file))
        data = []
        
        for record in fitfile.get_messages('record'):
            # Extract relevant fields - adjust based on available sensors
            record_data = {}
            for data_field in record:
                if data_field.name in ['accelerometer_x', 'accelerometer_y', 'accelerometer_z']:
                    record_data[data_field.name] = data_field.value
            
            if record_data:  # Only append if we have accelerometer data
                data.append(record_data)
                
        return pd.DataFrame(data)
    
    def create_spectrogram(self, data, fs=50):
        """Convert IMU data window to spectrogram image"""
        # Create spectrograms for each axis
        f, t, Sxx_x = signal.spectrogram(data['accelerometer_x'], fs=fs)
        f, t, Sxx_y = signal.spectrogram(data['accelerometer_y'], fs=fs)
        f, t, Sxx_z = signal.spectrogram(data['accelerometer_z'], fs=fs)
        
        # Combine into RGB image
        spectrogram = np.stack([Sxx_x, Sxx_y, Sxx_z], axis=2)
        
        # Normalize
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        
        return spectrogram
    
    def segment_and_label(self, df, activity_name):
        """Segment data into windows and create images"""
        windows = []
        
        for i in range(0, len(df) - WINDOW_SIZE, OVERLAP):
            window = df.iloc[i:i + WINDOW_SIZE]
            spec = self.create_spectrogram(window)
            
            # Save spectrogram as image
            img_path = self.processed_path/f"{activity_name}_{i}.png"
            plt.imsave(str(img_path), spec)
            
            windows.append({
                'image': str(img_path),
                'activity': activity_name
            })
            
        return windows

    def process_all_files(self):
        """Process all FIT files in data directory"""
        all_data = []
        
        # Assumes directory structure: data_path/activity_name/fit_files
        for activity in ACTIVITIES:
            activity_path = self.data_path/activity
            if not activity_path.exists():
                continue
                
            for fit_file in activity_path.glob('*.fit'):
                print(f"Processing {fit_file}")
                df = self.load_fit_file(fit_file)
                windows = self.segment_and_label(df, activity)
                all_data.extend(windows)
        
        # Create dataset CSV
        pd.DataFrame(all_data).to_csv(self.data_path/'dataset.csv', index=False)
        return self.data_path/'dataset.csv'

class GarminHAR:
    def __init__(self, data_csv):
        self.data_csv = data_csv
        
    def setup_dataloaders(self):
        """Set up FastAI dataloaders - similar to lecture 2"""
        df = pd.read_csv(self.data_csv)
        
        # Create FastAI DataLoaders (similar to lecture 2)
        dls = ImageDataLoaders.from_df(df,
                                     valid_pct=0.2,
                                     seed=42,
                                     label_col='activity',
                                     item_tfms=Resize(224),
                                     batch_tfms=aug_transforms())
        return dls
    
    def train_model(self, dls):
        """Train the model - similar to lecture 2"""
        learn = vision_learner(dls, resnet18, metrics=accuracy)
        learn.fine_tune(3)
        return learn
    
    def count_reps(self, df, activity):
        """Basic rep counting using peak detection"""
        if activity == 'deadlift':
            # Look at vertical acceleration
            peaks, _ = signal.find_peaks(df['accelerometer_y'], 
                                       height=1.5,
                                       distance=50)
        elif activity == 'kb_press':
            # Look at forward acceleration
            peaks, _ = signal.find_peaks(df['accelerometer_x'],
                                       height=1.0,
                                       distance=50)
        else:
            return 0
            
        return len(peaks)

def main():
    # Example usage
    data_path = Path('path/to/your/garmin/data')
    
    # Process raw Garmin data
    processor = GarminDataProcessor(data_path)
    dataset_csv = processor.process_all_files()
    
    # Train model
    har = GarminHAR(dataset_csv)
    dls = har.setup_dataloaders()
    learn = har.train_model(dls)
    
    # Example prediction on new data
    new_fit_file = data_path/'new_workout.fit'
    df = processor.load_fit_file(new_fit_file)
    
    # Make predictions and count reps
    # Note: You'll need to implement the prediction logic based on windows
    activity = learn.predict(window_spectrogram)[0]
    reps = har.count_reps(df, activity)
    print(f"Detected activity: {activity} with {reps} repetitions")

if __name__ == '__main__':
    main()