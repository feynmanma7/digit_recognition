<h1>Digit Recognition</h1>

# Requirements

> tensorflow: 2.0+

> numpy

> pillow, Image, for Image processing.

# Train model
## Base CNN
Put `mnist.npz` in `~/.keras/datasets`.

> python src/train_base_cnn.py

# Test model
## Base CNN
> python src/test_base_rnn.py image_path

# Notes
Currently, `8` cannot be easily recognized.
