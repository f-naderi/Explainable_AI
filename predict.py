# predict.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import load_mnist_from_github


def predict_from_test_set(model, X_test, y_test, num_samples=5, save_path=None):
    """Test predictions on random samples from the test set."""
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        prediction = np.argmax(model.predict(X_test[idx:idx+1], verbose=0))
        true_label = y_test[idx]
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[idx], cmap='gray')
        color = 'green' if prediction == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {prediction}", color=color, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle("Predictions on Test Set", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Predictions saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def main():
    print("=" * 50)
    print("MNIST Prediction Script")
    print("=" * 50)
    
    # Load model
    try:
        model = tf.keras.models.load_model('model/mnist_model.keras')
        print("Model loaded successfully.")
    except:
        print("ERROR: Model not found. Run train.py first.")
        sys.exit(1)
    
    # Load test data
    (_, _), (X_test, y_test) = load_mnist_from_github()
    X_test = X_test / 255.0
    
    # Run predictions on random test samples
    predict_from_test_set(model, X_test, y_test, num_samples=8, save_path='images/test.png')

if __name__ == "__main__":
    main()