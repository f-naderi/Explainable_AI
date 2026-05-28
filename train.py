# train.py
import tensorflow as tf
from model import create_model
from utils import load_mnist_from_github, plot_training_history, enable_gpu_memory_growth

def main():
    print("=" * 50)
    print("MNIST Training Script")
    print("=" * 50)
    
    enable_gpu_memory_growth()
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist_from_github()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Image size: {X_train.shape[1]}x{X_train.shape[2]}")
    print("-" * 50)
    
    # Create model
    model = create_model()
    model.summary()
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )
    
    print("-" * 50)
    print("Training completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    
    # Save model
    model.save('model/mnist_model.keras')
    print("\nModel saved as 'mnist_model.keras'")
    
    # Plot history
    plot_training_history(history, save_path='images/training_history.png')
    
    

if __name__ == "__main__":
    main()