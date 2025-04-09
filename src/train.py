import os
from data_loader import SatelliteDataLoader
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model():
    # Initialize data loader
    data_dir = 'training_testing_data'
    img_size = (224, 224)
    data_loader = SatelliteDataLoader(data_dir, img_size)
    
    # Load training and validation data
    X_train, y_train, _ = data_loader.load_train_data()
    X_test, y_test, _ = data_loader.load_test_data()
    
    # Create and compile model
    model = create_model(input_shape=(*img_size, 3))
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stopping]
    )
    
    return history, model

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    history, model = train_model()
    
    # Save the final model
    model.save('models/final_model.h5')