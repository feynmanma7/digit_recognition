from tensorflow.keras import optimizers, losses
from train_base_cnn import build_model
from utils import load_preprocess_data
import tensorflow as tf
from PIL import Image
import numpy as np
import sys


def load_model():
    model = build_model()

    optimizer = optimizers.Adam(learning_rate=0.001)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    (x_train, y_train), (x_test, y_test) = load_preprocess_data()

    checkpoint_dir = 'models/base_cnn/'
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    model.load_weights(checkpoint)

    return model


def load_image(image_path=None):
    img = Image.open(image_path).convert('L') # L: 8-bit, gray
    img = img.resize((28, 28))
    img = 255 - np.array(img) # revert the black and white pixel
    img = img/255 # normalize
    img = img[np.newaxis, :, :, np.newaxis] # [batch_size=1, 28, 28, channels=1]
    return img


def test_model(image_path=None):
    model = load_model()
    img = load_image(image_path=image_path)
    print('img.shape', img.shape)
    predictions = model.predict(img)

    print('Predictions:', predictions)
    print('Probability rank:', np.argsort(-predictions))
    print('\n\nThe digit is: ', np.argmax(predictions), '\n\n\n')


if __name__ == '__main__':
    image_path = '../images/5.png'
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    test_model(image_path=image_path)