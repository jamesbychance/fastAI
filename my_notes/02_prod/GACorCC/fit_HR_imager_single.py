from fitparse import FitFile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import pandas as pd

def save_exact_size_image(data, output_path, size=(224, 224)):
    """Save image with exact pixel dimensions"""
    # Create figure with exact pixel size
    dpi = 100
    figsize = (size[0]/dpi, size[1]/dpi)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Add axis that fills entire figure
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display image and remove axes
    ax.imshow(data, cmap='viridis', interpolation='nearest')
    ax.axis('off')
    
    # Save with exact dimensions
    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close()
    
    # Verify size
    from PIL import Image
    with Image.open(output_path) as img:
        if img.size != size:
            print(f"Warning: Output image size {img.size} doesn't match target size {size}")

def create_heatmap(heart_rates, output_path, size=(224, 224)):
    """Create a 224x224 heatmap visualization"""
    # Create matrix of exact size
    matrix = np.zeros(size)
    
    # Interpolate heart rate data
    n_samples = len(heart_rates)
    for i in range(size[0]):
        for j in range(size[1]):
            data_idx = int((i * size[1] + j) * n_samples / (size[0] * size[1]))
            if data_idx < n_samples:
                matrix[i, j] = heart_rates[data_idx]
    
    # Normalize
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    save_exact_size_image(matrix, output_path, size)

def create_spectrogram(heart_rates, output_path, size=(224, 224)):
    """Create a 224x224 spectrogram visualization"""
    # Calculate spectrogram
    nperseg = min(256, len(heart_rates)//4)
    noverlap = nperseg//2
    
    frequencies, times, Sxx = signal.spectrogram(
        heart_rates,
        fs=1.0,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum'
    )
    
    # Take log and normalize
    Sxx = np.log1p(Sxx)
    
    # Resize to exact dimensions using numpy
    from scipy.ndimage import zoom
    zoom_h = size[0] / Sxx.shape[0]
    zoom_w = size[1] / Sxx.shape[1]
    Sxx_resized = zoom(Sxx, (zoom_h, zoom_w))
    
    # Normalize
    Sxx_resized = (Sxx_resized - np.min(Sxx_resized)) / (np.max(Sxx_resized) - np.min(Sxx_resized))
    save_exact_size_image(Sxx_resized, output_path, size)

def create_recurrence_plot(heart_rates, output_path, size=(224, 224)):
    """Create a 224x224 recurrence plot visualization"""
    # Normalize heart rates
    normalized = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
    
    # Create distance matrix
    N = len(normalized)
    threshold = 0.1
    
    # Use broadcasting for efficient computation
    X = normalized.reshape(N, 1)
    Y = normalized.reshape(1, N)
    recurrence = np.abs(X - Y) < threshold
    
    # Resize to exact dimensions
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
    
    # Calculate minutes from start
    df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
    # Extract desired duration
    if take_last:
        max_minutes = df['minutes'].max()
        start_minute = max_minutes - duration_minutes
        df = df[df['minutes'] >= start_minute].copy()
    else:
        df = df[df['minutes'] <= duration_minutes].copy()
    
    # Reset minutes to start at 0
    df['minutes'] = df['minutes'] - df['minutes'].min()
    
    return df

def process_workout(file_path, output_dir):
    """Process a single workout file and create all visualizations"""
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract heart rate data
    print(f"\nProcessing: {file_path.name}")
    df = extract_heart_rate(file_path)
    heart_rates = df['heart_rate'].values
    
    # Create visualizations
    base_name = file_path.stem
    
    print("Creating heatmap...")
    create_heatmap(heart_rates, output_dir / f"{base_name}_heatmap.png")
    
    print("Creating spectrogram...")
    create_spectrogram(heart_rates, output_dir / f"{base_name}_spectrogram.png")
    
    print("Creating recurrence plot...")
    create_recurrence_plot(heart_rates, output_dir / f"{base_name}_recurrence.png")
    
    # Verify final sizes
    for suffix in ['heatmap', 'spectrogram', 'recurrence']:
        img_path = output_dir / f"{base_name}_{suffix}.png"
        from PIL import Image
        with Image.open(img_path) as img:
            print(f"{suffix.capitalize()} size: {img.size}")
    
    print(f"\nProcessing complete for: {file_path.name}")
    print(f"Duration: {df['minutes'].max():.1f} minutes")
    print(f"Samples: {len(df)}")
    print(f"HR range: {df['heart_rate'].min():.0f} - {df['heart_rate'].max():.0f} bpm")

def main():
    file_path = input("Enter the path to your .fit file: ")
    output_dir = Path("processed_images")
    
    process_workout(file_path, output_dir)

if __name__ == "__main__":
    main()