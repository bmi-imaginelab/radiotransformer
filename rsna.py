import tensorflow as tf

def decode_fn(record_bytes):
    return tf.io.parse_single_example(record_bytes,
      {"image": tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
       "label": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'x': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'y': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'w': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        'h': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
      })

def process_dataset(raw_dataset):
    image = tf.io.decode_raw(raw_dataset['image'], out_type=tf.uint8)
    image = tf.reshape(image, [256, 256])
    image = tf.cast(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = image / 255.
    label = raw_dataset['label']
    label = tf.one_hot(label, depth=2)
    label = tf.cast(label, dtype=tf.float64)
    return image, label

def rsna_pneumonia_data():
    full_ds = tf.data.TFRecordDataset("/home/mbhattac/project_radiotransformer/rsna_pneumonia.tfrecord").map(decode_fn)
    full_ds = full_ds.shuffle(buffer_size=10000, seed=1111)
    full_ds = full_ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    full_size = sum(1 for _ in full_ds)
    train_size = int(0.7 * full_size)
    val_size = int(0.1 * full_size)
    test_size = int(0.2 * full_size)
    print(train_size, val_size, test_size)
    train_dataset = full_ds.take(train_size)
    test_dataset = full_ds.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=1111)
    val_dataset = val_dataset.shuffle(buffer_size=10000, seed=1111)
    test_dataset = test_dataset.shuffle(buffer_size=10000, seed=1111)
    train_dataset = train_dataset.batch(batch_size=64, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size=64, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size=64, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = rsna_pneumonia_data()