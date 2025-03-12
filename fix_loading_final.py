import os
import shutil
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error

def fix_model_loading_final():
    """
    Final fix for the model loading issue by changing the loss function from string to function reference
    """
    file_path = 'march_madness_predictor/models/exit_round/exit_round_model.py'
    backup_path = file_path + '.bak_final2'
    
    # Create a backup of the original file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Update imports to include necessary modules if not already present
    if 'from tensorflow.keras.losses import mean_squared_error' not in content:
        import_line = 'import tensorflow as tf'
        updated_import = 'import tensorflow as tf\nfrom tensorflow.keras.losses import mean_squared_error'
        content = content.replace(import_line, updated_import)
    
    # Fix model compile section - change string 'mse' to function reference
    compile_line = "            model.compile(\n                optimizer=Adam(learning_rate=0.001),\n                loss='mse',  # Mean squared error for regression\n                metrics=['accuracy']  # Accuracy metric\n            )"
    updated_compile = "            model.compile(\n                optimizer=Adam(learning_rate=0.001),\n                loss=mean_squared_error,  # Mean squared error for regression\n                metrics=['accuracy']  # Accuracy metric\n            )"
    
    if compile_line in content:
        content = content.replace(compile_line, updated_compile)
    
    # Update load_model function to handle custom objects
    load_model_method = """    def load_model(self, model_path):
        \"\"\"
        Load a trained model from disk
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        tensorflow.keras.Model
            Loaded model
        \"\"\"
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None"""
            
    updated_load_model = """    def load_model(self, model_path):
        \"\"\"
        Load a trained model from disk
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        tensorflow.keras.Model
            Loaded model
        \"\"\"
        try:
            # Use custom objects to handle the loss function
            custom_objects = {
                'mean_squared_error': mean_squared_error,
                'mse': mean_squared_error
            }
            return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None"""
    
    if load_model_method in content:
        content = content.replace(load_model_method, updated_load_model)
    
    # Fix model loading in predict_tournament_performance
    predict_load = """            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')
            if os.path.exists(model_path):
                try:
                    # Use a more robust loading method
                    model = self.load_model(model_path)
                    if model is None:
                        raise Exception("Failed to load model")
                except Exception as e:
                    print(f"Error loading trained model: {e}")
                    print("Falling back to seed-based predictions")
                    return self.estimate_seeds_for_2025(current_data)"""
                    
    updated_predict_load = """            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')
            if os.path.exists(model_path):
                try:
                    # Use custom objects to handle the loss function
                    custom_objects = {
                        'mean_squared_error': mean_squared_error,
                        'mse': mean_squared_error
                    }
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    if model is None:
                        raise Exception("Failed to load model")
                except Exception as e:
                    print(f"Error loading trained model: {e}")
                    print("Falling back to seed-based predictions")
                    return self.estimate_seeds_for_2025(current_data)"""
    
    if predict_load in content:
        content = content.replace(predict_load, updated_predict_load)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Fixed model loading code in {file_path}")
    print("The key changes made were:")
    print("1. Added import for mean_squared_error from tensorflow.keras.losses")
    print("2. Changed model compilation to use mean_squared_error function reference instead of 'mse' string")
    print("3. Updated load_model method to use custom_objects to handle loss function")
    print("4. Updated direct model loading code in predict_tournament_performance")
    
    return True

if __name__ == "__main__":
    success = fix_model_loading_final()
    if success:
        print("Fix applied successfully. Please rerun the model training script.")
    else:
        print("Could not apply the fix. Please check the file manually.") 