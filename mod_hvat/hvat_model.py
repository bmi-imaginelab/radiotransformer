import tensorflow as tf

from hvat_model_utils import *

patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 32  # Initial image size

num_patch_x = 256 // patch_size[0]
num_patch_y = 256 // patch_size[1]

def hvat_teacher_model():
    inputs = tf.keras.layers.Input((256, 256, 3))
    # x = tf.keras.layers.RandomCrop(256, 256)(inputs)
    # x = ltf.keras.ayers.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(inputs)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    
    x_f_1_t = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=4,
    window_size=2,
    shift_size=0,
    num_mlp=128,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_f_1_t'
    )(x)
    
    x_s_1_t = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=2,
    window_size=1,
    shift_size=0,
    num_mlp=64,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_1_t'
    )(x)
    
    x_s_2_t = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=4,
    window_size=2,
    shift_size=shift_size,
    num_mlp=128,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_2_t'
    )(x_s_1_t)
    
    x_skip_t = tf.keras.layers.Add()([x_f_1_t, x_s_2_t])
    ema_t = tf.train.ExponentialMovingAverage(decay=0.0001)
    ema_t.average(x_skip_t)
    
    x_f_2_t = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=8,
    window_size=4,
    shift_size=shift_size,
    num_mlp=256,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_f_2_t'
    )(x_skip_t)
    
    x_s_3_t = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=4,
    window_size=2,
    shift_size=shift_size+1,
    num_mlp=128,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_3_t'
    )(x_skip_t)
    
    x_s_4_t = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=8,
    window_size=4,
    shift_size=shift_size+2,
    num_mlp=256,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_4_t'
    )(x_s_3_t)
    
    x_t = tf.keras.layers.Add()([x_f_2_t, x_s_4_t])
    
    x_t = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x_t)
    x_t = tf.keras.layers.GlobalAveragePooling1D()(x_t)
    
    class_output_t = tf.keras.layers.Dense(3, activation="softmax")(x_t)
    bbox_output_t = tf.keras.layers.Dense(10, activation="relu")(x_t)
    bbox_output_t = tf.keras.layers.Dense(4, activation="sigmoid")(bbox_output_t)
    
    model_t = tf.keras.Model(inputs=[inputs], outputs=[class_output_t, bbox_output_t])
    return model_t

if __name__ == '__main__':
    hvat_teacher_model()
