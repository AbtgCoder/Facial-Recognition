# using https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf paper

from cgi import test
from tabnanny import check
import uuid
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# Avoid out of memory errors, by settting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    # print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)


# Setup paths
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negative")
ANC_PATH = os.path.join("data", "anchor")

# make the directories
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# Move LFW Images to the following repository data/negative
# for directory in os.listdir("lfw"):
#     for file in os.listdir(os.path.join("lfw", directory)):
#         EX_PATH = os.path.join("lfw", directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)


# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()

#     frame = frame[120 : 120 + 2250, 200 : 200 + 250, :]
#     cv2.imshow("Image collection", frame)

#     # collect anchors
#     if cv2.waitKey(1) & 0xFF == ord("a"):
#         imgname = os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")
#         cv2.imwrite(imgname, frame)

#     # collect positives
#     if cv2.waitKey(1) & 0xFF == ord("p"):
#         imgname = os.path.join(POS_PATH, f"{uuid.uuid1()}.jpg")
#         cv2.imwrite(imgname, frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


anchor = tf.data.Dataset.list_files(ANC_PATH + "\*.jpg").take(300)
positive = tf.data.Dataset.list_files(POS_PATH + "\*.jpg").take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + "\*.jpg").take(300)


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)  # read img from file
    img = tf.io.decode_jpeg(byte_img)  # load the img
    img = tf.image.resize(img, (100, 100))  # resize
    img = img / 255.0  # normalize
    return img


# Create labeled dataset
# (anchor, positive) => 1, 1, 1, 1, 1
# (anchor, negative) => 0, 0, 0, 0, 0
positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))
)
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
)
data = positives.concatenate(negatives)


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# Build dataLoader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


# Training partition
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def make_embedding():
    inp = Input(shape=(100, 100, 3))

    # First Block
    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

    # Second Block
    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

    # Third Block
    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding")


embedding = make_embedding()
# embedding.sumnmary()


class L1Dist(Layer):  # create a custom distance layer
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model():

    # Anchor img input in the network
    input_image = Input(name="input_img", shape=(100, 100, 3))
    # Validation img input in the network
    validation_image = Input(name="validation_img", shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification Layer
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name="SiameseNetwork",
    )


siamese_model = make_siamese_model()
# siamese_model.summary()


binary_cross_loss = tf.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001


checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:

        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)

        # Calculate loss
        loss = binary_cross_loss(y, yhat)

    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


def train(data, EPOCHS):
    # Loop through EPOCHS

    for epoch in range(1, EPOCHS + 1):
        print(f"\n Epoch {epoch}/{EPOCHS}")
        progbar = tf.keras.utils.Progbar(len(data))

        # loop through each batch
        for idx, batch in enumerate(data):
            # run train step here
            train_step(batch)

            progbar.update(idx + 1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50


# train(train_data, EPOCHS)

# siamese_model.save("siamese_model.h5")

# test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# Reload model
model = tf.keras.models.load_model(
    "siamese_model.h5",
    custom_objects={
        "L1Dist": L1Dist,
        "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
    },
)
# print(model.predict([test_input, test]))

# Import metric calculations
# from tensorflow.keras.metrics import Precision, Recall


# # Make Predictions

# y_hat = model.predict([test_input, test_val])

# res = []
# for prediction in y_hat:
#     if prediction > 0.5:
#         res.append(1)
#     else:
#         res.append(0)

# m = Recall()
# n = Precision()
# m.update_state(y_true, y_hat)
# n.update_state(y_true, y_hat)

# print("Precision:", n.result().numpy()) # approx : 0.85
# print("Recall:", m.result().numpy())  # approx : 1.0
