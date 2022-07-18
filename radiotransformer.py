import tensorflow as tf

from radiotransformer_utils import *

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

def teacher_model():
    inputs = tf.keras.layers.Input((256, 256, 3))
    
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
    
    x_class_t = tf.keras.layers.Dense(3, activation="softmax")(x_t)
    bbox_output_t = tf.keras.layers.Dense(10, activation="relu")(x_t)
    bbox_output_t = tf.keras.layers.Dense(4, activation="sigmoid")(bbox_output_t)
    
    model_t = tf.keras.Model(inputs=inputs, outputs=[x_class_t, bbox_output_t, x_skip_t])
    return model_t

def student_model():
    inputs = tf.keras.layers.Input((256, 256, 3))
    
    x = PatchExtract(patch_size)(inputs)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    
    x_f_1_s = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=4,
    window_size=2,
    shift_size=0,
    num_mlp=128,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_f_1_s'
    )(x)
    
    x_s_1_s = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=2,
    window_size=1,
    shift_size=0,
    num_mlp=64,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_1_s'
    )(x)
    
    x_s_2_s = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=4,
    window_size=2,
    shift_size=shift_size,
    num_mlp=128,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_2_s'
    )(x_s_1_s)
    
    x_skip_s = tf.keras.layers.Add()([x_f_1_s, x_s_2_s])
    ema_s = tf.train.ExponentialMovingAverage(decay=0.0001)
    ema_s.average(x_skip_s)
    
    x_f_2_s = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=8,
    window_size=4,
    shift_size=shift_size,
    num_mlp=256,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_f_2_s'
    )(x_skip_s)
    
    x_s_3_s = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=4,
    window_size=2,
    shift_size=shift_size+1,
    num_mlp=128,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_3_s'
    )(x_skip_s)
    
    x_s_4_s = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=8,
    window_size=4,
    shift_size=shift_size+2,
    num_mlp=256,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
    prefix=f'layers_s_4_s'
    )(x_s_3_s)
    
    x_s = tf.keras.layers.Add()([x_f_2_s, x_s_4_s])
    
    x_s = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x_s)
    x_s = tf.keras.layers.GlobalAveragePooling1D()(x_s)
    
    model_s = tf.keras.Model(inputs=inputs, outputs=[x_s, x_skip_s])
    return model_s

def loaded_teacher_model():
    inputs = tf.keras.layers.Input((256, 256, 3))
    model_x = teacher_model()
    # model_x.load_weights('saved_checkpoint/main_pretrain_best_0.hdf5')
    model_loaded_t = tf.keras.Model(inputs=[model_x.input], outputs=[model_x.layers[-4].output, model_x.layers[-10].output])
    return model_loaded_t

def student_teacher_model():
    inputs = tf.keras.layers.Input((256, 256, 3))
    # class_teacher, skip_teacher = teacher_model()(inputs)
    class_teacher, skip_teacher = loaded_teacher_model()(inputs)
    class_student, skip_student = student_model()(inputs)
    
    skip_final = tf.keras.layers.Add()([skip_teacher, skip_student])
    ema_skip = tf.train.ExponentialMovingAverage(decay=0.0001)
    ema_skip.average(skip_final)
    
    class_final = tf.keras.layers.Add()([class_teacher, class_student])
    ema_class = tf.train.ExponentialMovingAverage(decay=0.0001)
    ema_class.average(class_final)
    
    class_final = tf.keras.layers.Dense(2, activation="softmax")(class_final)
    
    bbox_output_s = tf.keras.layers.Dense(10, activation="relu")(class_student)
    bbox_output_s = tf.keras.layers.Dense(4, activation="sigmoid")(bbox_output_s)
    
    bbox_output_t = tf.keras.layers.Dense(10, activation="relu")(class_teacher)
    bbox_output_t = tf.keras.layers.Dense(4, activation="sigmoid")(bbox_output_t)
    
    bbox = tf.stack([bbox_output_s, bbox_output_t])
    
    model_final = tf.keras.Model(inputs=[inputs], outputs=[class_final, bbox])
    
    return model_final

if __name__ == '__main__':
    student_teacher_model()
