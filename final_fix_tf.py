import os
import shutil
import tensorflow as tf

def fix_model_final():
    """
    Final fix for the model loading issue by properly using TensorFlow's MSE loss
    """
    file_path = 'march_madness_predictor/models/exit_round/exit_round_model.py'
    backup_path = file_path + '.bak_final3'
    
    # Create a backup of the original file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Modify the imports - replacing sklearn's MSE with TensorFlow's
    if 'from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error' in content:
        content = content.replace(
            'from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error',
            'from sklearn.metrics import confusion_matrix, classification_report'
        )
    
    # Add TensorFlow MSE import if not present
    if 'from tensorflow.keras.losses import MeanSquaredError' not in content:
        tf_import = 'import tensorflow as tf'
        updated_tf_import = 'import tensorflow as tf\nfrom tensorflow.keras.losses import MeanSquaredError'
        content = content.replace(tf_import, updated_tf_import)
    
    # Change model.compile to use the proper 'mse' string (which TensorFlow understands)
    compile_old = """            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=mean_squared_error,  # Mean squared error for regression
                metrics=['accuracy']  # Accuracy metric
            )"""
            
    compile_new = """            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',  # Mean squared error for regression
                metrics=['accuracy']  # Accuracy metric
            )"""
    
    if compile_old in content:
        content = content.replace(compile_old, compile_new)
    
    # Fix the load_model method to use custom objects
    load_model_old = """    def load_model(self, model_path):
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
    
    load_model_new = """    def load_model(self, model_path):
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
            # Use TensorFlow's built-in MSE loss
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None"""
    
    if load_model_old in content:
        content = content.replace(load_model_old, load_model_new)
    
    # Fix the predict_tournament_performance method
    predict_load_old = """            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')
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
    
    predict_load_new = """            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')
            if os.path.exists(model_path):
                try:
                    # Use TensorFlow's built-in loading capability
                    model = tf.keras.models.load_model(model_path)
                    if model is None:
                        raise Exception("Failed to load model")
                except Exception as e:
                    print(f"Error loading trained model: {e}")
                    print("Falling back to seed-based predictions")
                    return self.estimate_seeds_for_2025(current_data)"""
    
    if predict_load_old in content:
        content = content.replace(predict_load_old, predict_load_new)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Fixed model loading code in {file_path}")
    print("The key changes made were:")
    print("1. Removed sklearn's mean_squared_error import")
    print("2. Changed model compilation to use the built-in 'mse' string which TensorFlow understands")
    print("3. Simplified model loading to use TensorFlow's built-in support for 'mse'")
    
    return True

if __name__ == "__main__":
    success = fix_model_final()
    if success:
        print("Fix applied successfully. Please rerun the model training script.")
    else:
        print("Could not apply the fix. Please check the file manually.") 