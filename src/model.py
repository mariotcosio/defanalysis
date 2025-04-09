from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a CNN model for satellite imagery classification
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of classes to predict
    Returns:
        Model: Compiled Keras model
    """
    # Use ResNet50V2 as base model (transfer learning)
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create new model on top
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model