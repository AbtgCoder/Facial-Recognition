import os
import numpy as np
import cv2
import tensorflow as tf

from main_program import L1Dist


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)  # read img from file
    img = tf.io.decode_jpeg(byte_img)  # load the img
    img = tf.image.resize(img, (100, 100))  # resize
    img = img / 255.0  # normalize
    return img


model = tf.keras.models.load_model(
    "siamese_model.h5",
    custom_objects={
        "L1Dist": L1Dist,
        "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
    },
)


# Verification Function
def verify(model, detection_threshold, verification_threshold):

    # Detection Threshold: Metric above whicch a prediction is considered positive
    # Verification Threshold: Proportion of positive predictions / total positive samples

    # Build results array
    results = []
    for image in os.listdir(os.path.join("application_data", "verification_images")):
        input_img = preprocess(
            os.path.join("application_data", "input_image", "input_image.jpg")
        )
        validation_img = preprocess(
            os.path.join("application_data", "verification_images", image)
        )

        # Make predictions
        result = model.predict(
            list(np.expand_dims([validation_img, validation_img], axis=1))
        )
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(
        os.listdir(os.path.join("application_data", "verification_images"))
    )
    verified = verification > verification_threshold

    return results, verified


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow("Verification", frame)

    frame = frame[120 : 120 + 2250, 200 : 200 + 250, :]
    if cv2.waitKey(10) & 0xFF == ord("v"):
        cv2.imwrite(
            os.path.join("application_data", "input_image", "input_image.jpg"), frame
        )

        # run verification
        results, verified = verify(model, 0.5, 0.5)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
