# tensorflow: 2.1.0
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from utils import load_preprocess_data


def build_model():
    # inputs: [batch_size, height, width, num_channels], default `channels last`
    # inputs: [None, 28, 28, 1]
    height = 28
    width = 28
    num_channels = 1
    inputs = layers.Input(shape=(height, width, num_channels))

    # conv_1: [None, 26, 26, 32=filters=output_size]
    # out = (in - K + 2P) / S + 1 = (28 - 3 + 2*0) / 1 + 1 = 26
    conv_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid',
                           activation='relu', name='conv_1')(inputs)

    # pool_1: [None, 13, 13, 32]
    pool_1 = layers.MaxPool2D(pool_size=(2, 2), padding='valid',
                              name='pool_1')(conv_1)

    # conv_2: [None, 11, 11, 64]
    # (13 - 3 + 2*0) / 1 + 1 = 11
    conv_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid',
                           activation='relu', name='conv_2')(pool_1)

    # pool_2: [None, 5, 5, 64]
    pool_2 = layers.MaxPool2D(pool_size=(2, 2), padding='valid',
                              name='pool_2')(conv_2)

    # conv_3: [None, 3, 3, 64]
    # (5 - 3 + 2*0) / 1 + 1 = 3
    conv_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid',
                           activation='relu', name='conv_3')(pool_2)

    # flatten: [None, 576=3*3*64]
    flatten = layers.Flatten(name='flatten')(conv_3)

    # fc_1: [None, 64]
    fc_1 = layers.Dense(units=64, activation='relu', name='fc_1')(flatten)

    # dropout: [None, 64]
    dropout_rate = 0.2
    dropout = layers.Dropout(rate=dropout_rate, seed=0, name='dropout')(fc_1)

    # softmax: [None, 10]
    num_class = 10
    softmax = layers.Dense(units=num_class, activation='softmax', name='softmax')(dropout)

    model = models.Model(inputs=inputs, outputs=softmax, name='mnist_cnn')

    return model


def transfer(data):
    return data[0], data[1]


def convert_to_data_set(x, y, repeat_times=None,
                        shuffle_buffer_size=None, batch_size=None):
    x_tensor = tf.convert_to_tensor(x)
    y_tensor = tf.convert_to_tensor(y)
    data_set = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))\
        .repeat(repeat_times)\
        .shuffle(shuffle_buffer_size)\
        .batch(batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # .map(transfer, num_parallel_calls=tf.data.experimental.AUTOTUNE)\

    return data_set


def train_model():
    model = build_model()
    print(model.summary())

    optimizer = optimizers.Adam(learning_rate=0.001)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    (x_train, y_train), (x_test, y_test) = load_preprocess_data()

    epochs = 10
    n_train = 60000
    n_test = 10000
    batch_size = 32
    steps_per_epoch = n_train // batch_size
    validation_steps = n_test // batch_size

    train_data_set = convert_to_data_set(x_train, y_train, repeat_times=epochs,
                                  shuffle_buffer_size=n_train,
                                  batch_size=batch_size)

    val_data_set = convert_to_data_set(x_test, y_test, repeat_times=epochs,
                                  shuffle_buffer_size=n_test,
                                  batch_size=batch_size)

    my_callbacks = []
    early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5, restore_best_weights=True)
    my_callbacks.append(early_stopping_cb)

    tensorboard_cb = callbacks.TensorBoard(log_dir='logs')
    my_callbacks.append(tensorboard_cb)

    checkpoint_path = 'models/base_cnn/ckpt'
    checkpoint_cb = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              save_best_only=True)
    my_callbacks.append(checkpoint_cb)

    history = model.fit(train_data_set,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_data_set,
              validation_steps=validation_steps,
              callbacks=my_callbacks)

    print('\n\n')
    train_result = model.evaluate(x_train, y_train)
    format_result(train_result, name='train')

    val_result = model.evaluate(x_test, y_test)
    format_result(val_result, name='val')

    return history


def format_result(result, name=None):
    loss = result[0]
    acc = result[1]
    print('%s_loss=%.4f, %s_acc=%.4f' % (name, loss, name, acc))


if __name__ == '__main__':
    history = train_model()