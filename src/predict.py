import numpy as np
from tensorflow.keras.models import load_model
from data_loader import SatelliteDataLoader

def predict_image(model, image_path, img_size=(224, 224)):
    """
    Predict deforestation probability for a single image
    Args:
        model: Loaded Keras model
        image_path (str): Path to the image file
        img_size (tuple): Size of input images expected by the model
    Returns:
        tuple: (predicted_class, probability)
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    probability = prediction[0][predicted_class]
    
    classes = ['Normal', 'Deforestation']
    return classes[predicted_class], probability

def evaluate_test_set():
    """
    Evaluate the model on the test set
    """
    # Load the trained model
    model = load_model('models/best_model.h5')
    
    # Load test data
    data_loader = SatelliteDataLoader('training_testing_data')
    X_test, y_test, coordinates = data_loader.load_test_data()
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f'Test Set Accuracy: {accuracy * 100:.2f}%')
    
    # Print detailed results
    classes = ['Normal', 'Deforestation']
    for i, (pred, true, coord) in enumerate(zip(predicted_classes, true_classes, coordinates)):
        print(f'Image {coord}:')
        print(f'  Predicted: {classes[pred]}')
        print(f'  Actual: {classes[true]}')
        print(f'  Confidence: {predictions[i][pred]:.2f}')
        print()

if __name__ == "__main__":
    evaluate_test_set()