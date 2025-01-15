from fitparse import FitFile
from datetime import datetime
import json
from pathlib import Path

def dump_fit_to_text(file_path, output_path=None):
    """
    Parse a .fit file and dump all its contents to a text file
    """
    fitfile = FitFile(file_path)
    
    if output_path is None:
        output_path = Path(file_path).with_suffix('.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"FIT File Analysis: {Path(file_path).name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Track message counts
        message_counts = {}
        
        # First pass: Count message types
        for message in fitfile.messages:
            message_counts[message.name] = message_counts.get(message.name, 0) + 1
        
        # Write summary
        f.write("Message Type Summary:\n")
        f.write("-" * 40 + "\n")
        for msg_type, count in message_counts.items():
            f.write(f"{msg_type}: {count} messages\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Second pass: Write all messages
        for message in fitfile.messages:
            # Message header
            f.write(f"\nMessage Type: {message.name}\n")
            f.write("-" * 40 + "\n")
            
            # Get all fields
            for field in message.fields:
                value = field.value
                
                # Format datetime objects
                if isinstance(value, datetime):
                    value = value.isoformat()
                # Format tuples/lists
                elif isinstance(value, (tuple, list)):
                    value = json.dumps(value)
                
                f.write(f"{field.name}: {value}\n")
            
            f.write("\n")  # Extra newline between messages

if __name__ == "__main__":
    file_path = str(input("Please enter path/to/your/fit/file.fit: "))
    # file_path = input("Enter the path to your .fit file: ")
    output_path = Path(file_path).with_suffix('.txt')
    dump_fit_to_text(file_path, output_path)
    print(f"\nOutput written to: {output_path}")