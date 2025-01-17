from fitparse import FitFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def explore_unknown_fields(file_path):
    """
    Analyze unknown fields in record messages for patterns suggesting IMU data
    """
    fitfile = FitFile(file_path)
    
    # Store all records first
    all_records = []
    print("Collecting record data...")
    
    for record in fitfile.get_messages('record'):
        record_data = record.get_values()
        all_records.append(record_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Identify unknown columns
    unknown_cols = [col for col in df.columns if str(col).startswith('unknown_')]
    
    print(f"\nFound {len(unknown_cols)} unknown fields in {len(df)} records")
    print("\n=== Unknown Fields Analysis ===")
    
    for column in unknown_cols:
        # Get non-null values
        values = df[column].dropna().values
        
        if len(values) == 0:
            print(f"\n{column}: All null values")
            continue
        
        # Basic statistics
        stats = {
            'count': len(values),
            'null_count': df[column].isna().sum(),
            'unique_values': len(pd.unique(values)),
            'min': np.min(values) if len(values) > 0 else None,
            'max': np.max(values) if len(values) > 0 else None,
            'mean': np.mean(values) if len(values) > 0 else None,
            'std': np.std(values) if len(values) > 0 else None
        }
        
        print(f"\n{column}:")
        print(f"- Sample size: {stats['count']} (Nulls: {stats['null_count']})")
        print(f"- Unique values: {stats['unique_values']}")
        print(f"- Range: {stats['min']} to {stats['max']}")
        if stats['mean'] is not None:
            print(f"- Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        # Check for potential IMU characteristics
        if len(values) > 0:
            # Check if values oscillate around zero
            zero_crossings = np.where(np.diff(np.signbit(values)))[0]
            
            # Check if values are in typical accelerometer range (-16g to +16g)
            in_accel_range = (-16 <= stats['min'] <= 16) and (-16 <= stats['max'] <= 16)
            
            # Check for regular sampling (consistent time differences)
            time_diffs = pd.Series(df['timestamp']).diff().dropna()
            regular_sampling = time_diffs.std().total_seconds() < 1.0
            
            # Calculate value changes
            value_changes = np.diff(values)
            rapid_changes = np.any(np.abs(value_changes) > 0.1)
            
            imu_likelihood = []
            if len(zero_crossings) > len(values)/100:  # More than 1% zero crossings
                imu_likelihood.append("oscillates around zero")
            if in_accel_range:
                imu_likelihood.append("within typical accelerometer range")
            if regular_sampling:
                imu_likelihood.append("regularly sampled")
            if rapid_changes:
                imu_likelihood.append("shows rapid value changes")
            
            if imu_likelihood:
                print("- IMU Characteristics:", ", ".join(imu_likelihood))
                
                # Plot time series and histogram side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
                
                # Time series plot (first 1000 points)
                plot_data = values[:1000]
                timestamps = df['timestamp'][:1000]
                relative_time = [(t - timestamps[0]).total_seconds() for t in timestamps]
                
                ax1.plot(relative_time, plot_data)
                ax1.set_title(f"{column} - First 1000 samples")
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Value")
                ax1.grid(True)
                
                # Histogram
                ax2.hist(values, bins=50, density=True)
                ax2.set_title(f"{column} - Value Distribution")
                ax2.set_xlabel("Value")
                ax2.set_ylabel("Density")
                ax2.grid(True)
                
                plt.tight_layout()
                plt.show()
                
                # Print additional statistical information
                print("- Additional Statistics:")
                print(f"  * Zero crossings: {len(zero_crossings)}")
                print(f"  * Average change between samples: {np.mean(np.abs(value_changes)):.4f}")
                print(f"  * Max change between samples: {np.max(np.abs(value_changes)):.4f}")

if __name__ == "__main__":
    file_path = str(input("Please enter path/to/your/fit/file.fit: "))
    # file_path = "data/CardioClub/CardioClub_20240531_FRI.fit"
    explore_unknown_fields(file_path)


# from fitparse import FitFile
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def explore_unknown_fields(file_path):
#     """
#     Analyze unknown fields in record messages for patterns suggesting IMU data
#     """
#     fitfile = FitFile(file_path)
    
#     # Dictionary to store time series for each unknown field
#     unknown_fields_data = defaultdict(list)
#     timestamps = []
    
#     # Collect data from record messages
#     print("Collecting unknown field data from records...")
#     for record in fitfile.get_messages('record'):
#         data = record.get_values()
#         timestamp = data['timestamp']
        
#         # Store all unknown fields
#         for field_name, value in data.items():
#             if field_name.startswith('unknown_'):
#                 unknown_fields_data[field_name].append(value)
#         timestamps.append(timestamp)
    
#     # Convert to DataFrame for analysis
#     df = pd.DataFrame(unknown_fields_data, index=timestamps)
    
#     print("\n=== Unknown Fields Analysis ===")
#     for column in df.columns:
#         # Get non-null values
#         values = df[column].dropna().values
        
#         if len(values) == 0:
#             print(f"\n{column}: All null values")
#             continue
            
#         # Basic statistics
#         stats = {
#             'count': len(values),
#             'null_count': df[column].isna().sum(),
#             'unique_values': len(set(values)),
#             'min': np.min(values) if len(values) > 0 else None,
#             'max': np.max(values) if len(values) > 0 else None,
#             'mean': np.mean(values) if len(values) > 0 else None,
#             'std': np.std(values) if len(values) > 0 else None
#         }
        
#         print(f"\n{column}:")
#         print(f"- Sample size: {stats['count']} (Nulls: {stats['null_count']})")
#         print(f"- Unique values: {stats['unique_values']}")
#         print(f"- Range: {stats['min']} to {stats['max']}")
#         print(f"- Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
#         # Check for potential IMU characteristics
#         if stats['count'] > 0:
#             # Check if values oscillate around zero
#             zero_crossings = np.where(np.diff(np.signbit(values)))[0]
            
#             # Check if values are in typical accelerometer range (-16g to +16g)
#             in_accel_range = (-16 <= stats['min'] <= 16) and (-16 <= stats['max'] <= 16)
            
#             # Check for regular sampling (consistent time differences)
#             time_diffs = pd.Series(timestamps).diff().dropna()
#             regular_sampling = time_diffs.std().total_seconds() < 1.0
            
#             imu_likelihood = []
#             if len(zero_crossings) > 0:
#                 imu_likelihood.append("oscillates around zero")
#             if in_accel_range:
#                 imu_likelihood.append("within typical accelerometer range")
#             if regular_sampling:
#                 imu_likelihood.append("regularly sampled")
            
#             if imu_likelihood:
#                 print("- IMU Characteristics:", ", ".join(imu_likelihood))
                
#                 # Plot the first 1000 points if it shows IMU characteristics
#                 plt.figure(figsize=(12, 4))
#                 plt.plot(values[:1000])
#                 plt.title(f"{column} - First 1000 samples")
#                 plt.xlabel("Sample")
#                 plt.ylabel("Value")
#                 plt.grid(True)
#                 plt.show()

# if __name__ == "__main__":
#     file_path = str(input("Please enter path/to/your/fit/file.fit: "))
#     # file_path = "CardioClub_20240531_FRI.fit"  # Replace with your file path
#     explore_unknown_fields(file_path)