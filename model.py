# model.py
import tensorflow as tf



def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 14×14
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),  # 14×14 ← heatmap 14×14!
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_model(model_type='cnn', input_shape=(28, 28, 1), num_classes=10):
    """
    Factory function to create different model architectures.
    
    Args:
        model_type: 'cnn' or 'dense'
        input_shape: Shape of input images
        num_classes: Number of output classes
    """
    if model_type == 'cnn':
        return create_cnn_model(input_shape, num_classes)
    elif model_type == 'dense':
        # Original simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

if __name__ == "__main__":
    model = create_model('cnn')
    model.summary()