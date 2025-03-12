import os
import shutil

def fix_neural_network():
    """
    Comprehensive fix for the exit round model training and prediction issues
    """
    file_path = 'march_madness_predictor/models/exit_round/exit_round_model.py'
    backup_path = file_path + '.bak_final'
    
    # Create a backup of the original file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # More comprehensive fix for model training and loading
    # 1. Update train_model method to fix the training error
    train_model_code = """    def train_model(self, prepared_data, epochs=100, batch_size=32):
        \"\"\"
        Train the model to predict tournament exit rounds
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data for training
        epochs : int, optional
            Number of epochs to train for
        batch_size : int, optional
            Batch size for training
            
        Returns:
        --------
        tensorflow.keras.Model
            Trained model
        \"\"\"
        print("\\nTraining tournament exit round prediction model...")
        
        # Get data from prepared_data
        X_scaled = prepared_data['X_scaled']
        y = prepared_data['y']
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Build the model
        model = self.build_model(input_shape=X_train.shape[1])
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'exit_round_model.keras'),
                monitor='val_loss',
                save_best_only=True
            ),
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )"""
        
    updated_train_model = """    def train_model(self, prepared_data, epochs=100, batch_size=32):
        \"\"\"
        Train the model to predict tournament exit rounds
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data for training
        epochs : int, optional
            Number of epochs to train for
        batch_size : int, optional
            Batch size for training
            
        Returns:
        --------
        tensorflow.keras.Model
            Trained model
        \"\"\"
        print("\\nTraining tournament exit round prediction model...")
        
        try:
            # Get data from prepared_data
            X_scaled = prepared_data['X_scaled']
            y = prepared_data['y']
            
            # Convert to numpy arrays if needed
            X_scaled = np.array(X_scaled)
            y = np.array(y)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Build the model
            input_dim = X_train.shape[1]
            model = Sequential([
                # Input layer
                Dense(128, activation='relu', input_dim=input_dim),
                BatchNormalization(),
                Dropout(0.3),
                
                # Hidden layers
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                # Output layer - regression for exit round (can be fractional)
                Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',  # Mean squared error for regression
                metrics=['accuracy']  # Accuracy metric
            )
            
            # Print model summary
            model.summary()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_save_path, 'exit_round_model.keras'),
                    monitor='val_loss',
                    save_best_only=True
                ),
            ]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )"""
    
    # Replace train_model method
    if train_model_code in content:
        content = content.replace(train_model_code, updated_train_model)
    
    # Remove build_model method since we're now building the model directly in train_model
    build_model_code = """    def build_model(self, input_shape):
        \"\"\"
        Build a neural network model for predicting tournament exit rounds
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Neural network model
        \"\"\"
        print("\\nBuilding neural network model...")
        
        # Create model
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer - regression for exit round (can be fractional)
            Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=mean_squared_error,  # Mean squared error for regression
            metrics=['accuracy']  # Accuracy metric
        )
        
        # Print model summary
        model.summary()
        
        return model"""
        
    # 2. Fix the load_model function to properly handle custom objects
    load_model_code = """    def load_model(self, model_path):
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
            # Load model with custom objects to handle loss function
            custom_objects = {
                'mean_squared_error': mean_squared_error,
                'mse': mean_squared_error
            }
            return load_model(model_path, custom_objects=custom_objects)
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
            # Use tf.keras.models.load_model with custom objects to handle loss function
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None"""
    
    # Replace load_model function
    if load_model_code in content:
        content = content.replace(load_model_code, updated_load_model)
    
    # 3. Fix predict_tournament_performance method to correctly load the model
    predict_load = """        # Load trained model
        if model is None:
            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')
            if os.path.exists(model_path):
                try:
                    # Load model with custom objects to handle loss function
                    custom_objects = {
                        'mean_squared_error': mean_squared_error,
                        'mse': mean_squared_error
                    }
                    model = load_model(model_path, custom_objects=custom_objects)
                except Exception as e:
                    print(f"Error loading trained model: {e}")
                    print("Falling back to seed-based predictions")
                    return self.estimate_seeds_for_2025(current_data)"""
                    
    updated_predict_load = """        # Load trained model
        if model is None:
            model_path = os.path.join(self.model_save_path, 'exit_round_model.keras')
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
    
    # Make replacements
    if predict_load in content:
        content = content.replace(predict_load, updated_predict_load)
    
    # Remove the build_model method since we're now building the model directly in train_model
    if build_model_code in content:
        content = content.replace(build_model_code, "")
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Fixed model code in {file_path}")
    
    return True

if __name__ == "__main__":
    success = fix_neural_network()
    if success:
        print("Comprehensive fix applied successfully. Please rerun the model training script.")
    else:
        print("Could not apply the fix. Please check the file manually.") 