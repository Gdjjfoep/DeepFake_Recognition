import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_convlstm_model(seq_length, img_size, num_classes):
    # Input Shape
    video_input = layers.Input(shape=(seq_length, img_size, img_size, 3))
    
    # 1. GPU Data Augmentation (Happens inside the GPU memory)
    x = layers.TimeDistributed(layers.RandomFlip("horizontal"))(video_input)
    x = layers.TimeDistributed(layers.RandomRotation(0.1))(x)
    x = layers.TimeDistributed(layers.RandomZoom(0.1))(x)

    # 2. EfficientNetB0 (Better than MobileNet)
    base_cnn = applications.EfficientNetB0(
        weights="imagenet", 
        include_top=False, 
        input_shape=(img_size, img_size, 3)
    )
    
    # 3. Fine-Tuning Strategy
    base_cnn.trainable = True
    # Freeze the early layers (generic shapes), unfreeze the top layers (specific textures)
    for layer in base_cnn.layers[:-30]:
        layer.trainable = False 

    # 4. Connect everything
    x = layers.TimeDistributed(base_cnn)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    
    # 5. LSTM (Temporal Logic)
    x = layers.LSTM(128)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x) # Higher dropout to prevent overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(video_input, outputs)
    return model