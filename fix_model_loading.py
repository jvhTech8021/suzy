import os
import fileinput
import shutil
import sys

def fix_model_loading():
    """
    Fixes the model loading issue by properly handling the 'mse' loss function
    """
    file_path = 'march_madness_predictor/models/exit_round/exit_round_model.py'
    backup_path = file_path + '.bak3'
    
    # Create a backup of the original file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Update imports to include necessary modules
    import_line = 'import tensorflow as tf'
    updated_import = 'import tensorflow as tf\nfrom tensorflow.keras.losses import mean_squared_error'
    
    # Add import if not already present
    if import_line in content and 'mean_squared_error' not in content:
        content = content.replace(import_line, updated_import)
    
    # Fix model compile section
    compile_line = "        model.compile(\n            optimizer=Adam(learning_rate=0.001),\n            loss='mse',  # Mean squared error for regression\n            metrics=['accuracy']  # Accuracy metric\n        )"
    updated_compile = "        model.compile(\n            optimizer=Adam(learning_rate=0.001),\n            loss=mean_squared_error,  # Mean squared error for regression\n            metrics=['accuracy']  # Accuracy metric\n        )"
    
    if compile_line in content:
        content = content.replace(compile_line, updated_compile)
    
    # Fix model loading code
    loading_code = "    def load_model(self, model_path):\n        \"\"\"\n        Load a trained model from disk\n        \n        Parameters:\n        -----------\n        model_path : str\n            Path to the saved model\n            \n        Returns:\n        --------\n        tensorflow.keras.Model\n            Loaded model\n        \"\"\"\n        try:\n            return load_model(model_path)\n        except Exception as e:\n            print(f\"Error loading model: {e}\")\n            return None"
    
    updated_loading = "    def load_model(self, model_path):\n        \"\"\"\n        Load a trained model from disk\n        \n        Parameters:\n        -----------\n        model_path : str\n            Path to the saved model\n            \n        Returns:\n        --------\n        tensorflow.keras.Model\n            Loaded model\n        \"\"\"\n        try:\n            # Load model with custom objects to handle loss function\n            custom_objects = {\n                'mean_squared_error': mean_squared_error,\n                'mse': mean_squared_error\n            }\n            return load_model(model_path, custom_objects=custom_objects)\n        except Exception as e:\n            print(f\"Error loading model: {e}\")\n            return None"
    
    if loading_code in content:
        content = content.replace(loading_code, updated_loading)
    
    # Also update the part where it loads the trained model in the predict_tournament_performance method
    predict_load = "        # Load trained model\n        if model is None:\n            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')\n            if os.path.exists(model_path):\n                try:\n                    model = load_model(model_path)\n                except Exception as e:\n                    print(f\"Error loading trained model: {e}\")\n                    print(\"Falling back to seed-based predictions\")\n                    return self.estimate_seeds_for_2025(current_data)"
    
    updated_predict_load = "        # Load trained model\n        if model is None:\n            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')\n            if os.path.exists(model_path):\n                try:\n                    # Load model with custom objects to handle loss function\n                    custom_objects = {\n                        'mean_squared_error': mean_squared_error,\n                        'mse': mean_squared_error\n                    }\n                    model = load_model(model_path, custom_objects=custom_objects)\n                except Exception as e:\n                    print(f\"Error loading trained model: {e}\")\n                    print(\"Falling back to seed-based predictions\")\n                    return self.estimate_seeds_for_2025(current_data)"
    
    if predict_load in content:
        content = content.replace(predict_load, updated_predict_load)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Fixed model loading code in {file_path}")
    
    return True

if __name__ == "__main__":
    success = fix_model_loading()
    if success:
        print("Fix applied successfully. Please rerun the model training script.")
    else:
        print("Could not apply the fix. Please check the file manually.") 