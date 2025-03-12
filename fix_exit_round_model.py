import os
import fileinput
import shutil
import sys

def fix_model_code():
    """
    Fixes the input_shape parameter in the build_model method of exit_round_model.py
    """
    file_path = 'march_madness_predictor/models/exit_round/exit_round_model.py'
    backup_path = file_path + '.bak'
    
    # Create a backup of the original file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Define the search line and replacement
    search_line = '            Dense(128, activation=\'relu\', input_shape=input_shape),'
    replacement = '            Dense(128, activation=\'relu\', input_shape=(input_shape,)),'
    
    # Track if we made the replacement
    replacement_made = False
    
    # Read the file, make changes, and write back
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Make the replacement
    if search_line in content:
        new_content = content.replace(search_line, replacement)
        replacement_made = True
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(new_content)
        
        print(f"Fixed input_shape parameter in {file_path}")
    else:
        print(f"Could not find the line to replace in {file_path}")
    
    return replacement_made

if __name__ == "__main__":
    success = fix_model_code()
    if success:
        print("Fix applied successfully. Please rerun the model training script.")
    else:
        print("Could not apply the fix. Please check the file manually.") 