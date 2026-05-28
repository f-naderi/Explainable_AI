# utils.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import os
import requests
import gzip
import struct
import cv2



DATA_DIR = './mnist_data'
BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master"

def load_mnist_from_github(cache_dir=DATA_DIR):
    """
    Load dataset from GitHub. If files already exist in cache_dir,
    they are read from disk and no download is needed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    raw_data = {}
    for name, filename in files.items():
        filepath = os.path.join(cache_dir, filename)
        
        # Download only if the file does not exist
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from GitHub...")
            url = f"{BASE_URL}/{filename}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filepath}")
        else:
            print(f"Found {filename} on disk. Loading...")
        
        # Read file from disk    
        with gzip.open(filepath, 'rb') as f:
            raw_data[name] = f.read()

    
    def parse_images(raw):
        magic, num, rows, cols = struct.unpack('>IIII', raw[:16])
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(num, rows, cols)

    def parse_labels(raw):
        magic, num = struct.unpack('>II', raw[:8])
        return np.frombuffer(raw[8:], dtype=np.uint8)

    X_train = parse_images(raw_data['train_images'])
    y_train = parse_labels(raw_data['train_labels'])
    X_test = parse_images(raw_data['test_images'])
    y_test = parse_labels(raw_data['test_labels'])
    
    return (X_train, y_train), (X_test, y_test)

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2, marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Training', linewidth=2, marker='o')
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close()
    else:
        plt.show()



def standardize_input(data):
    """
    Takes any array-like input and converts it to a
    proper NumPy array with shape (n, 28, 28, 1).
    """
    # 1. Convert TensorFlow tensor to NumPy if needed  
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    
    # 2. Ensure float32 data type
    data = data.astype(np.float32)
    
    # 3. Reshape based on the number of dimensions  
    if data.ndim == 2:        # (28, 28) -> (1, 28, 28, 1)
        data = data[np.newaxis, ..., np.newaxis]
    elif data.ndim == 3:      # (n, 28, 28) -> (n, 28, 28, 1)
        # If the first dimension is greater than 1, it is a batch
        if data.shape[0] > 1:
            data = data[..., np.newaxis]
        else:                 # (1, 28, 28) -> (1, 28, 28, 1)
            data = data[..., np.newaxis]
    elif data.ndim == 4:      # (n, 28, 28, 1) -> OK, do nothing  
        pass
    else:
        raise ValueError(f"Unexpected number of dimensions: {data.ndim}")
        
    return data


    
def show_sample_predictions(model, X_test, y_test, num_samples=10, save_path=None):

    #X_test=standardize_input(X_test)
    
    rng=np.random.default_rng(seed=42)
    indices = rng.choice(len(X_test), num_samples, replace=False)
    probabilities = model.predict(X_test[indices], verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[idx], cmap='gray')
        
        color = 'green' if predictions[i] == y_test[idx] else 'red'
        plt.title(f"True: {y_test[idx]}\nPred: {predictions[i]}",
                  color=color, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle("Sample Predictions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Predictions saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {len(gpus)}")
    else:
        print("No GPU found. Running on CPU.")
    print("=" * 50)




# ============================================================
# Grad-CAM Functions
# ============================================================



def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1"):
    """Alternative approach: pause gradient after conv layer."""
    
    # Get the conv layer
    conv_layer = model.get_layer(last_conv_layer_name)
    
    # Find the index of the conv layer
    conv_layer_idx = None
    for i, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            conv_layer_idx = i
            break
    
    # Build a model that goes up to the conv layer
    partial_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.layers[conv_layer_idx].output
    )
    
    # Get conv features
    with tf.GradientTape() as tape:
        conv_outputs = partial_model(img_array)
        tape.watch(conv_outputs)
        
        # Continue through the rest of the layers manually
        x = conv_outputs
        for layer in model.layers[conv_layer_idx + 1:]:
            x = layer(x)
        predictions = x
        
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    
    grads = tape.gradient(loss, conv_outputs)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()





def explain_predictions(model, X_test, y_test, num_samples=5, save_path=None):
    """Show all Grad-CAM explanations in a single figure."""
    rng=np.random.default_rng(seed=42)
    indices = rng.choice(len(X_test), num_samples, replace=False)


    # Create a single figure with num_samples rows and 3 columns
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, 2 * num_samples))
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        img_array = np.expand_dims(img, axis=0)
        
        prob = model.predict(img_array, verbose=0)
        pred = np.argmax(prob)
                
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_3")
        
        # Resize and color heatmap
        heatmap_resized = cv2.resize(heatmap, (28, 28), interpolation=cv2.INTER_LINEAR)
        heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]
        img_rgb = np.stack([img] * 3, axis=-1)
        superimposed = heatmap_colored * 0.4 + img_rgb * 0.6
        
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Original\nTrue: {y_test[idx]}")
        axes[i, 0].axis('off')
        
        # Heatmap
        axes[i, 1].imshow(heatmap_resized, cmap='jet')
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(superimposed)
        color = 'green' if pred == y_test[idx] else 'red'
        axes[i, 2].set_title(f"Overlay\nPred: {pred}", color=color)
        axes[i, 2].axis('off')
    
    plt.suptitle("Grad-CAM: Where the Model Looks", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
        plt.close()
    else:
        plt.show()


