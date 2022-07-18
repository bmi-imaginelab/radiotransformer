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
    image = tf.io.decode_raw(raw_dataset['image'], out_type=tf.uint16)
    image = tf.reshape(image, [256, 256])
    image = tf.cast(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    
    #channel_avg = tf.constant([0.485, 0.456, 0.406])
    #channel_std = tf.constant([0.229, 0.224, 0.225])
    #image = (image / 255.0 - channel_avg) / channel_std
    #image = tf.image.resize(image, [224, 224])
    
    image = tf.image.stateless_random_brightness(image, max_delta=0.05, seed=(1, 1))
    image = tf.image.stateless_random_contrast(image, lower=1.0, upper=1.5, seed=(1, 1))
    image = tf.image.stateless_random_saturation(image, lower=1.0, upper=2.0, seed=(1, 1))
    image = image / 255.
    
    label = raw_dataset['label']
    label = tf.one_hot(label, depth=3)
    label = tf.cast(label, dtype=tf.float32)
    bbox = [tf.cast(raw_dataset['x'], dtype=tf.float32)/256., tf.cast(raw_dataset['y'], dtype=tf.float32)/256., tf.cast(raw_dataset['w'], dtype=tf.float32)/256., tf.cast(raw_dataset['h'], dtype=tf.float32)/256.]
    bbox = tf.convert_to_tensor(bbox, dtype=tf.float32)
    return image, (label, bbox)

def physionet1_data():
    full_ds = tf.data.TFRecordDataset("gs://imagine_lab/eye_gaze_mimic.tfrecord").map(decode_fn)
    full_ds = full_ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_len = sum(1 for _ in full_ds)
    train_size = int(0.7 * train_len)
    print(train_len, train_size)
    train_ds = full_ds.take(train_size)
    test_ds = full_ds.skip(train_size)
    train_ds = train_ds.shuffle(buffer_size=10000)#.repeat()
    train_ds = train_ds.batch(8, drop_remainder=True)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.shuffle(buffer_size=10000)#.repeat()
    test_ds = test_ds.batch(8, drop_remainder=True)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, test_ds

def physionet1_data_ablation(percentage=0.5):
    full_ds = tf.data.TFRecordDataset("gs://imagine_lab/eye_gaze_mimic.tfrecord").map(decode_fn)
    full_ds = full_ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_len = sum(1 for _ in full_ds)
    train_size = int(percentage * train_len)
    print(train_len, train_size)
    train_ds = full_ds.take(train_size)
    test_ds = full_ds.skip(train_size)
    train_ds = train_ds.shuffle(buffer_size=10000)#.repeat()
    train_ds = train_ds.batch(8, drop_remainder=True)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.shuffle(buffer_size=10000)#.repeat()
    test_ds = test_ds.batch(8, drop_remainder=True)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, test_ds

if __name__ == '__main__':
    physionet1_data_ablation(percentage=0.6)