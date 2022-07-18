import tensorflow as tf
import tensorflow_addons as tfa

def visual_attention_loss(y_true, y_pred):
    student_pred, teacher_pred = tf.unstack(y_pred)
    g_iou = tfa.losses.GIoULoss(mode = 'giou', reduction = tf.keras.losses.Reduction.SUM)
    mse = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.SUM)
    
    final = 0.2 * g_iou(student_pred, teacher_pred) + 0.8 * mse(student_pred, teacher_pred)
    
    return tf.reduce_mean(final)

if __name__ == '__main__':
    flag = 0