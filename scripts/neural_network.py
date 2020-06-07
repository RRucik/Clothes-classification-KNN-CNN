import tensorflow as tf
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0


model = keras.Sequential([

    keras.layers.Convolution2D(32, (3, 3), input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.Activation('relu'),
    keras.layers.Convolution2D(32, (3, 3)),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10)

])


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
model.summary()