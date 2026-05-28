# evaluate.py
import tensorflow as tf
from utils import load_mnist_from_github, show_sample_predictions, explain_predictions
import numpy as np


def main():
    print("=" * 50)
    print("MNIST Evaluation Script")
    print("=" * 50)
    
    # Load saved model
    model = tf.keras.models.load_model('model/mnist_model.keras')
    print("Model loaded from 'mnist_model.keras'")
    
    # Load test data
    _, (X_test, y_test) = load_mnist_from_github()
    X_test = X_test / 255.0
    print(f"Test samples: {X_test.shape[0]:,}")
    print("-" * 50)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("=" * 50)
    print("Final Results on 10,000 Test Images:")
    print(f"   Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Correct predictions: {int(test_accuracy * len(X_test)):,} out of {len(X_test):,}")
    print("=" * 50)
    
    # Show predictions
    show_sample_predictions(model, X_test[:10], y_test[:10], save_path='images/predictions.png')


    # Explain predictions
    explain_predictions(model, X_test[:10], y_test[:10], num_samples=5, save_path='images/Grad_CAM.png')



if __name__ == "__main__":
    main()