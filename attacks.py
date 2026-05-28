# attacks.py
import tensorflow as tf
from utils import load_mnist_from_github, show_sample_predictions, explain_predictions
import numpy as np

def fgsm_attack(model, images, labels, epsilon=0.1):
    images = tf.cast(images, tf.float32)
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=-1)
    
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    
    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
    
    return tf.squeeze(adversarial_images)


def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, steps=40):
    """
    PGD Attack for a batch of images - simple version.
    """
    original_images = tf.identity(images)
    adversarial_images = tf.identity(images)
    
    # Add channel dimension if 3D 
    if len(adversarial_images.shape) == 3:
        adversarial_images = tf.expand_dims(adversarial_images, axis=-1)
        original_images = tf.expand_dims(original_images, axis=-1)
    
    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_images)
            predictions = model(adversarial_images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        
        gradient = tape.gradient(loss, adversarial_images)
        
        # Small step in the gradient direction  
        adversarial_images = adversarial_images + alpha * tf.sign(gradient)
        
        # Clip noise  
        noise = adversarial_images - original_images
        noise = tf.clip_by_value(noise, -epsilon, epsilon)
        adversarial_images = original_images + noise

        # Keep within [0, 1] range
        adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
    
    return tf.squeeze(adversarial_images)


    
def main():
    # Load saved model
    model = tf.keras.models.load_model('model/mnist_model.keras')
    print("Model loaded from 'mnist_model.keras'")
    
    # Load test data
    _, (X_test, y_test) = load_mnist_from_github()
    X_test = X_test[:10] / 255.0

    # Test FGSM
    adversarial_fgsm = fgsm_attack(model, X_test, y_test[:10], epsilon=0.15)
    


    show_sample_predictions(model, adversarial_fgsm.numpy(), y_test[:10], save_path='images/FGSM_predictions.png')
    explain_predictions(model, adversarial_fgsm.numpy(), y_test[:10], num_samples=5, save_path='images/Grad_CAM_FGSM.png')
    
    # Test PGD 
    adversarial_pgd = pgd_attack(model, X_test, y_test[:10], epsilon=0.15, alpha=0.01875, steps=20)
    


    show_sample_predictions(model, adversarial_pgd.numpy(), y_test[:10], save_path='images/PGD_predictions.png')
    explain_predictions(model, adversarial_pgd.numpy(), y_test[:10], num_samples=5, save_path='images/Grad_CAM_PGD.png')
    

if __name__ == "__main__":
    main()