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


# --- ---- --- ---- ----- ---

# from fitparse import FitFile
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from pathlib import Path
# import pandas as pd

# """3 types of visualisation:

# Heatmap:

# High-resolution grid matching ResNet input size
# Smooth interpolation of heart rate data


# Spectrogram:

# Optimised segment size and overlap
# Log scaling for better pattern visibility
# Properly scaled to 224x224


# Recurrence Plot:

# Efficient matrix computation using broadcasting
# Proper resizing to 224x224
# Binary patterns preserved during resizing

# """

# def extract_heart_rate(file_path, duration_minutes=45, take_last=True):
#     """Extract heart rate data from fit file"""
#     fitfile = FitFile(str(file_path))
#     data = []
    
#     for record in fitfile.get_messages('record'):
#         values = record.get_values()
#         if 'heart_rate' in values and 'timestamp' in values:
#             data.append({
#                 'timestamp': values['timestamp'],
#                 'heart_rate': values['heart_rate']
#             })
    
#     df = pd.DataFrame(data)
#     df = df.sort_values('timestamp')
    
#     # Calculate minutes from start
#     df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
#     # Extract desired duration
#     if take_last:
#         max_minutes = df['minutes'].max()
#         start_minute = max_minutes - duration_minutes
#         df = df[df['minutes'] >= start_minute].copy()
#     else:
#         df = df[df['minutes'] <= duration_minutes].copy()
    
#     # Reset minutes to start at 0
#     df['minutes'] = df['minutes'] - df['minutes'].min()
    
#     return df

# def save_resnet_format(data, output_path, size=(224, 224)):
#     """Save data as a 224x224 RGB image using viridis colormap"""
#     plt.figure(figsize=(8, 8))
#     plt.imshow(data, cmap='viridis')
#     plt.axis('off')
#     plt.savefig(output_path, dpi=28, bbox_inches='tight', pad_inches=0)
#     plt.close()

# def create_heatmap(heart_rates, output_path, size=(224, 224)):
#     """Create a 224x224 heatmap visualisation"""
#     # Create a high-resolution grid
#     grid_size = size
#     matrix = np.zeros(grid_size)
    
#     # Interpolate heart rate data to fill the grid
#     n_samples = len(heart_rates)
#     for i in range(grid_size[0]):
#         for j in range(grid_size[1]):
#             data_idx = int((i * grid_size[1] + j) * n_samples / (grid_size[0] * grid_size[1]))
#             if data_idx < n_samples:
#                 matrix[i, j] = heart_rates[data_idx]
    
#     # Normalize the data
#     matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    
#     # Save as image
#     save_resnet_format(matrix, output_path)

# def create_spectrogram(heart_rates, output_path, size=(224, 224)):
#     """Create a 224x224 spectrogram visualisation"""
#     # Calculate spectrogram with optimal parameters for visualisation
#     nperseg = min(256, len(heart_rates)//4)  # Adjust segment size based on data length
#     noverlap = nperseg//2
    
#     frequencies, times, Sxx = signal.spectrogram(
#         heart_rates,
#         fs=1.0,  # 1 Hz sampling rate (normalized)
#         nperseg=nperseg,
#         noverlap=noverlap,
#         scaling='spectrum'
#     )
    
#     # Take log of power spectrum for better visualisation
#     Sxx = np.log1p(Sxx)
    
#     # Normalize
#     Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
    
#     # Save as image
#     save_resnet_format(Sxx, output_path)

# def create_recurrence_plot(heart_rates, output_path, size=(224, 224)):
#     """Create a 224x224 recurrence plot visualization"""
#     # Normalise heart rates
#     normalized = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
    
#     # Create distance matrix
#     N = len(normalized)
#     threshold = 0.1  # Adjust this value to change pattern sensitivity
    
#     # Use broadcasting for efficient computation
#     X = normalized.reshape(N, 1)
#     Y = normalized.reshape(1, N)
#     recurrence = np.abs(X - Y) < threshold
    
#     # Resize to 224x224 using nearest neighbor interpolation
#     if N != size[0]:
#         from scipy.ndimage import zoom
#         zoom_factor = size[0] / N
#         recurrence = zoom(recurrence, zoom_factor, order=0)
    
#     # Save as image
#     save_resnet_format(recurrence, output_path)

# def process_workout(file_path, output_dir):
#     """Process a single workout file and create all visualizations"""
#     file_path = Path(file_path)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Extract heart rate data
#     print(f"\nProcessing: {file_path.name}")
#     df = extract_heart_rate(file_path)
#     heart_rates = df['heart_rate'].values
    
#     # Create visualizations
#     base_name = file_path.stem
    
#     print("Creating heatmap...")
#     create_heatmap(heart_rates, output_dir / f"{base_name}_heatmap.png")
    
#     print("Creating spectrogram...")
#     create_spectrogram(heart_rates, output_dir / f"{base_name}_spectrogram.png")
    
#     print("Creating recurrence plot...")
#     create_recurrence_plot(heart_rates, output_dir / f"{base_name}_recurrence.png")
    
#     # Print summary
#     print(f"\nProcessing complete for: {file_path.name}")
#     print(f"Duration: {df['minutes'].max():.1f} minutes")
#     print(f"Samples: {len(df)}")
#     print(f"HR range: {df['heart_rate'].min():.0f} - {df['heart_rate'].max():.0f} bpm")
#     print(f"Output saved to: {output_dir}")

# def main():
#     # Get file path from user
#     file_path = input("Enter the path to your .fit file: ")
#     output_dir = Path("processed_images")
    
#     process_workout(file_path, output_dir)

# if __name__ == "__main__":
#     main()

# --- --- --- 

# from fitparse import FitFile
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import pandas as pd
# from pathlib import Path

# def extract_heart_rate(file_path, duration_minutes=45, take_last=True):
#     """Extract heart rate data from fit file"""
#     # Convert Path to string for FitFile
#     fitfile = FitFile(str(file_path))
#     data = []
    
#     for record in fitfile.get_messages('record'):
#         values = record.get_values()
#         if 'heart_rate' in values and 'timestamp' in values:
#             data.append({
#                 'timestamp': values['timestamp'],
#                 'heart_rate': values['heart_rate']
#             })
    
#     df = pd.DataFrame(data)
#     df = df.sort_values('timestamp')
    
#     # Calculate minutes from start
#     df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
#     # Extract desired duration
#     if take_last:
#         max_minutes = df['minutes'].max()
#         start_minute = max_minutes - duration_minutes
#         df = df[df['minutes'] >= start_minute].copy()
#     else:
#         df = df[df['minutes'] <= duration_minutes].copy()
    
#     # Reset minutes to start at 0
#     df['minutes'] = df['minutes'] - df['minutes'].min()
    
#     return df

# def save_visualizations(heart_rates, sampling_rate, output_dir, base_name):
#     """Create and save all four visualizations for the heart rate data"""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # 1. Heatmap (15x15 grid)
#     plt.figure(figsize=(8, 8))
#     matrix_size = (15, 15)
#     n_samples = len(heart_rates)
#     samples_per_cell = n_samples // (matrix_size[0] * matrix_size[1])
#     matrix = np.zeros(matrix_size)
#     for i in range(matrix_size[0]):
#         for j in range(matrix_size[1]):
#             idx = (i * matrix_size[1] + j) * samples_per_cell
#             if idx + samples_per_cell <= len(heart_rates):
#                 matrix[i, j] = np.mean(heart_rates[idx:idx + samples_per_cell])
    
#     plt.imshow(matrix, cmap='viridis')
#     plt.title('Heatmap Representation')
#     plt.colorbar(label='Heart Rate (bpm)')
#     plt.savefig(output_dir / f"{base_name}_heatmap.png")
#     plt.close()
    
#     # 2. Spectrogram
#     plt.figure(figsize=(8, 8))
#     frequencies, times, Sxx = signal.spectrogram(
#         heart_rates,
#         fs=sampling_rate,
#         nperseg=len(heart_rates)//20,
#         noverlap=len(heart_rates)//40
#     )
#     plt.imshow(np.log1p(Sxx), aspect='auto', cmap='viridis')
#     plt.title('Spectrogram')
#     plt.colorbar(label='Log Power')
#     plt.savefig(output_dir / f"{base_name}_spectrogram.png")
#     plt.close()
    
#     # 3. Multi-channel
#     plt.figure(figsize=(8, 8))
#     # Channel 1: Normalized heart rates
#     ch1 = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
#     # Channel 2: Rate of change (first derivative)
#     ch2 = np.gradient(heart_rates)
#     ch2 = (ch2 - np.min(ch2)) / (np.max(ch2) - np.min(ch2))
#     # Channel 3: Acceleration (second derivative)
#     ch3 = np.gradient(ch2)
#     ch3 = (ch3 - np.min(ch3)) / (np.max(ch3) - np.min(ch3))
    
#     multi_channel = np.stack([ch1, ch2, ch3], axis=-1)
#     plt.imshow(multi_channel)
#     plt.title('Multi-channel Representation\n(Red: HR, Green: Rate of Change, Blue: Acceleration)')
#     plt.savefig(output_dir / f"{base_name}_multichannel.png")
#     plt.close()
    
#     # 4. Recurrence Plot
#     plt.figure(figsize=(8, 8))
#     n = len(heart_rates)
#     normalized = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
#     matrix = np.zeros((n, n))
#     threshold = 0.1
    
#     # Use vectorization for faster computation
#     X, Y = np.meshgrid(normalized, normalized)
#     matrix = np.abs(X - Y) < threshold
    
#     plt.imshow(matrix, cmap='binary')
#     plt.title('Recurrence Plot')
#     plt.colorbar(label='Recurrence')
#     plt.savefig(output_dir / f"{base_name}_recurrence.png")
#     plt.close()
    
#     print(f"\nVisualizations saved in: {output_dir}")

# def main():
#     # Get file path from user
#     file_path = input("Enter the path to your .fit file: ")
#     file_path = Path(file_path)
    
#     # Create output directory
#     output_dir = Path('visualizations')
    
#     # Extract heart rate data
#     print("Extracting heart rate data...")
#     df = extract_heart_rate(file_path)
    
#     # Calculate sampling rate (Hz)
#     sampling_rate = len(df) / df['minutes'].max() / 60
    
#     # Create and save visualizations
#     print("Creating visualizations...")
#     save_visualizations(
#         df['heart_rate'].values, 
#         sampling_rate, 
#         output_dir, 
#         file_path.stem
#     )
    
#     # Print some basic stats
#     print("\nData Summary:")
#     print(f"Duration: {df['minutes'].max():.1f} minutes")
#     print(f"Number of samples: {len(df)}")
#     print(f"Sampling rate: {sampling_rate:.2f} Hz")
#     print(f"Average heart rate: {df['heart_rate'].mean():.1f} bpm")
#     print(f"Min heart rate: {df['heart_rate'].min():.1f} bpm")
#     print(f"Max heart rate: {df['heart_rate'].max():.1f} bpm")

# if __name__ == "__main__":
#     main()


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# from fitparse import FitFile
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import pandas as pd
# from pathlib import Path

# def extract_heart_rate(file_path, duration_minutes=45, take_last=True):
#     """Extract heart rate data from fit file"""
#     fitfile = FitFile(file_path)
#     data = []
    
#     for record in fitfile.get_messages('record'):
#         values = record.get_values()
#         if 'heart_rate' in values and 'timestamp' in values:
#             data.append({
#                 'timestamp': values['timestamp'],
#                 'heart_rate': values['heart_rate']
#             })
    
#     df = pd.DataFrame(data)
#     df = df.sort_values('timestamp')
    
#     # Calculate minutes from start
#     df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
#     # Extract desired duration
#     if take_last:
#         max_minutes = df['minutes'].max()
#         start_minute = max_minutes - duration_minutes
#         df = df[df['minutes'] >= start_minute].copy()
#     else:
#         df = df[df['minutes'] <= duration_minutes].copy()
    
#     # Reset minutes to start at 0
#     df['minutes'] = df['minutes'] - df['minutes'].min()
    
#     return df

# def save_visualizations(heart_rates, sampling_rate, output_dir, base_name):
#     """Create and save all four visualizations for the heart rate data"""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # 1. Heatmap (15x15 grid)
#     plt.figure(figsize=(8, 8))
#     matrix_size = (15, 15)
#     n_samples = len(heart_rates)
#     samples_per_cell = n_samples // (matrix_size[0] * matrix_size[1])
#     matrix = np.zeros(matrix_size)
#     for i in range(matrix_size[0]):
#         for j in range(matrix_size[1]):
#             idx = (i * matrix_size[1] + j) * samples_per_cell
#             matrix[i, j] = np.mean(heart_rates[idx:idx + samples_per_cell])
    
#     plt.imshow(matrix, cmap='viridis')
#     plt.title('Heatmap Representation')
#     plt.colorbar(label='Heart Rate (bpm)')
#     plt.savefig(output_dir / f"{base_name}_heatmap.png")
#     plt.close()
    
#     # 2. Spectrogram
#     plt.figure(figsize=(8, 8))
#     frequencies, times, Sxx = signal.spectrogram(
#         heart_rates,
#         fs=sampling_rate,
#         nperseg=len(heart_rates)//20,
#         noverlap=len(heart_rates)//40
#     )
#     plt.imshow(np.log1p(Sxx), aspect='auto', cmap='viridis')
#     plt.title('Spectrogram')
#     plt.colorbar(label='Log Power')
#     plt.savefig(output_dir / f"{base_name}_spectrogram.png")
#     plt.close()
    
#     # 3. Multi-channel
#     plt.figure(figsize=(8, 8))
#     # Channel 1: Normalized heart rates
#     ch1 = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
#     # Channel 2: Rate of change (first derivative)
#     ch2 = np.gradient(heart_rates)
#     ch2 = (ch2 - np.min(ch2)) / (np.max(ch2) - np.min(ch2))
#     # Channel 3: Acceleration (second derivative)
#     ch3 = np.gradient(ch2)
#     ch3 = (ch3 - np.min(ch3)) / (np.max(ch3) - np.min(ch3))
    
#     multi_channel = np.stack([ch1, ch2, ch3], axis=-1)
#     plt.imshow(multi_channel)
#     plt.title('Multi-channel Representation\n(Red: HR, Green: Rate of Change, Blue: Acceleration)')
#     plt.savefig(output_dir / f"{base_name}_multichannel.png")
#     plt.close()
    
#     # 4. Recurrence Plot
#     plt.figure(figsize=(8, 8))
#     n = len(heart_rates)
#     normalized = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
#     matrix = np.zeros((n, n))
#     threshold = 0.1
    
#     for i in range(n):
#         for j in range(n):
#             diff = abs(normalized[i] - normalized[j])
#             matrix[i, j] = 1 if diff < threshold else 0
    
#     plt.imshow(matrix, cmap='binary')
#     plt.title('Recurrence Plot')
#     plt.colorbar(label='Recurrence')
#     plt.savefig(output_dir / f"{base_name}_recurrence.png")
#     plt.close()
    
#     print(f"\nVisualizations saved in: {output_dir}")

# def main():
#     # Get file path from user
#     # file_path = input("Enter the path to your .fit file: ")
#     file_path = str(input("Please enter path/to/your/fit/file.fit: "))
#     file_path = Path(file_path)
    
#     # Create output directory
#     output_dir = Path('visualizations')
    
#     # Extract heart rate data
#     print("Extracting heart rate data...")
#     df = extract_heart_rate(file_path)
    
#     # Calculate sampling rate (Hz)
#     sampling_rate = len(df) / df['minutes'].max() / 60
    
#     # Create and save visualizations
#     print("Creating visualizations...")
#     save_visualizations(
#         df['heart_rate'].values, 
#         sampling_rate, 
#         output_dir, 
#         file_path.stem
#     )
    
#     # Print some basic stats
#     print("\nData Summary:")
#     print(f"Duration: {df['minutes'].max():.1f} minutes")
#     print(f"Number of samples: {len(df)}")
#     print(f"Sampling rate: {sampling_rate:.2f} Hz")
#     print(f"Average heart rate: {df['heart_rate'].mean():.1f} bpm")
#     print(f"Min heart rate: {df['heart_rate'].min():.1f} bpm")
#     print(f"Max heart rate: {df['heart_rate'].max():.1f} bpm")

# if __name__ == "__main__":
#     main()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# from fitparse import FitFile
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import pandas as pd

# def extract_heart_rate(file_path, duration_minutes=45, take_last=True):
#     """Extract heart rate data from fit file"""
#     fitfile = FitFile(file_path)
#     data = []
    
#     for record in fitfile.get_messages('record'):
#         values = record.get_values()
#         if 'heart_rate' in values and 'timestamp' in values:
#             data.append({
#                 'timestamp': values['timestamp'],
#                 'heart_rate': values['heart_rate']
#             })
    
#     df = pd.DataFrame(data)
#     df = df.sort_values('timestamp')
    
#     # Calculate minutes from start
#     df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
#     # Extract desired duration
#     if take_last:
#         max_minutes = df['minutes'].max()
#         start_minute = max_minutes - duration_minutes
#         df = df[df['minutes'] >= start_minute].copy()
#     else:
#         df = df[df['minutes'] <= duration_minutes].copy()
    
#     # Reset minutes to start at 0
#     df['minutes'] = df['minutes'] - df['minutes'].min()
    
#     return df

# def create_visualizations(heart_rates, sampling_rate):
#     """Create all four visualizations for the heart rate data"""
#     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
#     fig.suptitle('Heart Rate Data Representations', fontsize=16)
    
#     # 1. Heatmap (15x15 grid)
#     matrix_size = (15, 15)
#     n_samples = len(heart_rates)
#     samples_per_cell = n_samples // (matrix_size[0] * matrix_size[1])
#     matrix = np.zeros(matrix_size)
#     for i in range(matrix_size[0]):
#         for j in range(matrix_size[1]):
#             idx = (i * matrix_size[1] + j) * samples_per_cell
#             matrix[i, j] = np.mean(heart_rates[idx:idx + samples_per_cell])
    
#     axes[0, 0].imshow(matrix, cmap='viridis')
#     axes[0, 0].set_title('Heatmap Representation')
#     axes[0, 0].axis('off')
    
#     # 2. Spectrogram
#     frequencies, times, Sxx = signal.spectrogram(
#         heart_rates,
#         fs=sampling_rate,
#         nperseg=len(heart_rates)//20,
#         noverlap=len(heart_rates)//40
#     )
#     axes[0, 1].imshow(np.log1p(Sxx), aspect='auto', cmap='viridis')
#     axes[0, 1].set_title('Spectrogram')
#     axes[0, 1].axis('off')
    
#     # 3. Multi-channel (showing as RGB)
#     # Channel 1: Normalized heart rates
#     ch1 = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
#     # Channel 2: Rate of change (first derivative)
#     ch2 = np.gradient(heart_rates)
#     ch2 = (ch2 - np.min(ch2)) / (np.max(ch2) - np.min(ch2))
#     # Channel 3: Acceleration (second derivative)
#     ch3 = np.gradient(ch2)
#     ch3 = (ch3 - np.min(ch3)) / (np.max(ch3) - np.min(ch3))
    
#     # Reshape for visualization
#     multi_channel = np.stack([ch1, ch2, ch3], axis=-1)
#     axes[1, 0].imshow(multi_channel)
#     axes[1, 0].set_title('Multi-channel Representation')
#     axes[1, 0].axis('off')
    
#     # 4. Recurrence Plot
#     n = len(heart_rates)
#     normalized = (heart_rates - np.min(heart_rates)) / (np.max(heart_rates) - np.min(heart_rates))
#     matrix = np.zeros((n, n))
#     threshold = 0.1
    
#     for i in range(n):
#         for j in range(n):
#             diff = abs(normalized[i] - normalized[j])
#             matrix[i, j] = 1 if diff < threshold else 0
    
#     axes[1, 1].imshow(matrix, cmap='binary')
#     axes[1, 1].set_title('Recurrence Plot')
#     axes[1, 1].axis('off')
    
#     plt.tight_layout()
#     return fig

# def main():
#     # Get file path from user
#     # file_path = input("Enter the path to your .fit file: ")
#     file_path = str(input("Please enter path/to/your/fit/file.fit: "))
    
#     # Extract heart rate data
#     print("Extracting heart rate data...")
#     df = extract_heart_rate(file_path)
    
#     # Calculate sampling rate (Hz)
#     sampling_rate = len(df) / df['minutes'].max() / 60
    
#     # Create visualizations
#     print("Creating visualizations...")
#     fig = create_visualizations(df['heart_rate'].values, sampling_rate)
    
#     # Show plot
#     plt.show()
    
#     # Print some basic stats
#     print("\nData Summary:")
#     print(f"Duration: {df['minutes'].max():.1f} minutes")
#     print(f"Number of samples: {len(df)}")
#     print(f"Sampling rate: {sampling_rate:.2f} Hz")
#     print(f"Average heart rate: {df['heart_rate'].mean():.1f} bpm")

# if __name__ == "__main__":
#     main()