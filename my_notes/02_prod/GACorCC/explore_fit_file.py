from fitparse import FitFile
from pathlib import Path

""" This python program looks to find all the names of the types of data available in the specified .fit file
"""

def explore_fit_file(file_path):
    """
    Explore and print the structure of a .fit file
    """
    fitfile = FitFile(file_path)
    
    # Dictionary to store message types and their fields
    message_fields = {}
    
    print(f"\nExploring file: {Path(file_path).name}")
    print("=" * 50)
    
    # First pass: Collect all message types and their fields
    for message in fitfile.messages:
        msg_type = message.name
        
        if msg_type not in message_fields:
            message_fields[msg_type] = set()
            
        for field in message.fields:
            message_fields[msg_type].add(field.name)
    
    # Second pass: Print examples for each message type
    for msg_type in sorted(message_fields.keys()):
        print(f"\nMessage Type: {msg_type}")
        print("-" * 30)
        
        # Get first message of this type
        try:
            sample_msg = next(fitfile.get_messages(msg_type))
            print("Sample fields:")
            for field in sorted(message_fields[msg_type]):
                value = sample_msg.get_value(field)
                print(f"  {field}: {value}")
        except StopIteration:
            print("No messages of this type found")
        
        print(f"Total unique fields: {len(message_fields[msg_type])}")

# Example usage
if __name__ == "__main__":
    file_path = str(input("Please enter path/to/your/fit/file.fit: "))
    # file_path = "path/to/your/fit/file.fit"  # Replace with actual path
    explore_fit_file(file_path)