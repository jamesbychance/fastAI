from fitparse import FitFile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import pandas as pd

# [Previous helper functions remain the same]
def save_exact_size_image(data, output_path, size=(224, 224)):
    """Save image with exact pixel dimensions"""
    dpi = 100
    figsize = (size[0]/dpi, size[1]/dpi)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(data, cmap='viridis', interpolation='nearest')
    ax.axis('off')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close()
    
    from PIL import Image
    with Image.open(output_path) as img:
        if img.size != size:
            print(f"Warning: Output image size {img.size} doesn't match target size {size}")

def create_heatmap(heart_rates, output_path, size=(224, 224)):
    """Create a 224x224 heatmap visualization"""
    matrix = np.zeros(size)
    
    n_samples = len(heart_rates)
    for i in range(size[0]):
        for j in range(size[1]):
            data_idx = int((i * size[1] + j) * n_samples / (size[0] * size[1]))
            if data_idx < n_samples:
                matrix[i, j] = heart_rates[data_idx]
    
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    save_exact_size_image(matrix, output_path, size)

def create_spectrogram(heart_rates, output_path, size=(224, 224)):
    """Create a 224x224 spectrogram visualization"""
    nperseg = min(256, len(heart_rates)//4)
    noverlap = nperseg//2
    
    frequencies, times, Sxx = signal.spectrogram(
        heart_rates,
        fs=1.0,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum'
    )
    
    Sxx = np.log1p(Sxx)
    
    from scipy.ndimage import zoom
    zoom_h = size[0] / Sxx.shape[0]
    zoom_w = size[1] / Sxx.shape[1]
    Sxx_resized = zoom(Sxx, (zoom_h, zoom_w))
    
    Sxx_resized = (Sxx_resized - np.min(Sxx_resized)) / (np.max(Sxx_resized) - np.min(Sxx_resized))
    save_exact_size_image(Sxx_resized, output_path, size)

def create_recurrence_plot(heart_rates, output_path, size=(224, 224)):
    """Create a 224x224 recurrence plot visualization"""
    normalized = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
    
    N = len(normalized)
    threshold = 0.1
    
    X = normalized.reshape(N, 1)
    Y = normalized.reshape(1, N)
    recurrence = np.abs(X - Y) < threshold
    
    from scipy.ndimage import zoom
    zoom_factor = size[0] / N
    recurrence_resized = zoom(recurrence, (zoom_factor, zoom_factor), order=0)
    
    save_exact_size_image(recurrence_resized, output_path, size)

def extract_heart_rate(file_path, duration_minutes=45, take_last=True):
    """Extract heart rate data from fit file"""
    fitfile = FitFile(str(file_path))
    data = []
    
    for record in fitfile.get_messages('record'):
        values = record.get_values()
        if 'heart_rate' in values and 'timestamp' in values:
            data.append({
                'timestamp': values['timestamp'],
                'heart_rate': values['heart_rate']
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')
    
    df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
    if take_last:
        max_minutes = df['minutes'].max()
        start_minute = max_minutes - duration_minutes
        df = df[df['minutes'] >= start_minute].copy()
    else:
        df = df[df['minutes'] <= duration_minutes].copy()
    
    df['minutes'] = df['minutes'] - df['minutes'].min()
    
    return df

def process_workout(file_path, output_dir):
    """Process a single workout file and create all visualizations"""
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing: {file_path.name}")
    df = extract_heart_rate(file_path)
    heart_rates = df['heart_rate'].values
    
    base_name = file_path.stem
    
    print("Creating heatmap...")
    create_heatmap(heart_rates, output_dir / f"{base_name}_heatmap.png")
    
    print("Creating spectrogram...")
    create_spectrogram(heart_rates, output_dir / f"{base_name}_spectrogram.png")
    
    print("Creating recurrence plot...")
    create_recurrence_plot(heart_rates, output_dir / f"{base_name}_recurrence.png")
    
    for suffix in ['heatmap', 'spectrogram', 'recurrence']:
        img_path = output_dir / f"{base_name}_{suffix}.png"
        from PIL import Image
        with Image.open(img_path) as img:
            print(f"{suffix.capitalize()} size: {img.size}")
    
    print(f"\nProcessing complete for: {file_path.name}")
    print(f"Duration: {df['minutes'].max():.1f} minutes")
    print(f"Samples: {len(df)}")
    print(f"HR range: {df['heart_rate'].min():.0f} - {df['heart_rate'].max():.0f} bpm")

def batch_process_folder(input_folder, output_base_dir, take_last=True):
    """
    Process all .fit files in a folder
    
    Args:
        input_folder (str or Path): Path to folder containing .fit files
        output_base_dir (str or Path): Base directory for output
        take_last (bool): If True, process last 45 mins, if False, process first 45 mins
    """
    input_folder = Path(input_folder)
    output_base_dir = Path(output_base_dir)
    
    # Create time period subfolder
    time_period = "last45min" if take_last else "first45min"
    output_dir = output_base_dir / time_period
    
    # Create visualization-specific subdirectories
    viz_dirs = {
        'heatmap': output_dir / 'heatmaps',
        'spectrogram': output_dir / 'spectrograms',
        'recurrence': output_dir / 'recurrence_plots'
    }
    
    # Create all directories
    for dir_path in viz_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .fit files in the input folder
    fit_files = list(input_folder.glob("*.fit"))
    
    if not fit_files:
        print(f"No .fit files found in {input_folder}")
        return
    
    print(f"\nFound {len(fit_files)} .fit files to process")
    print(f"Processing {'last' if take_last else 'first'} 45 minutes of each file")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    for i, fit_file in enumerate(fit_files, 1):
        print(f"\nProcessing file {i} of {len(fit_files)}")
        try:
            df = extract_heart_rate(fit_file, duration_minutes=45, take_last=take_last)
            heart_rates = df['heart_rate'].values
            
            base_name = fit_file.stem
            
            # Create all visualizations
            for viz_type, create_func in [
                ('heatmap', create_heatmap),
                ('spectrogram', create_spectrogram),
                ('recurrence', create_recurrence_plot)
            ]:
                output_path = viz_dirs[viz_type] / f"{base_name}_{viz_type}.png"
                create_func(heart_rates, output_path)
            
            print(f"Successfully processed: {fit_file.name}")
            print(f"Duration: {df['minutes'].max():.1f} minutes")
            print(f"HR range: {df['heart_rate'].min():.0f} - {df['heart_rate'].max():.0f} bpm")
            
        except Exception as e:
            print(f"Error processing {fit_file.name}: {str(e)}")
            continue

def main():
    """Enhanced main function with batch processing options"""
    print("\nFIT File Batch Processor")
    print("------------------------")
    
    # Get input folder
    while True:
        input_folder = input("Enter the path to your folder containing .fit files: ")
        input_path = Path(input_folder)
        if input_path.exists() and input_path.is_dir():
            break
        print("Invalid folder path. Please try again.")
    
    # Get output directory
    output_dir = input("Enter the path for output directory (default: 'processed_images'): ").strip()
    if not output_dir:
        output_dir = "processed_images"
    
    # Get processing preference
    while True:
        choice = input("Process (F)irst or (L)ast 45 minutes of each workout? [F/L]: ").strip().upper()
        if choice in ['F', 'L']:
            break
        print("Invalid choice. Please enter 'F' for first 45 minutes or 'L' for last 45 minutes.")
    
    take_last = (choice == 'L')
    
    # Process the files
    batch_process_folder(input_path, output_dir, take_last)
    
    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()