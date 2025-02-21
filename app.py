import numpy as np
import gradio as gr
import keras_cv
from PIL import Image
from transformers import pipeline
from tensorflow.keras.optimizers import AdamW # type: ignore

class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']
model = keras_cv.models.ImageClassifier.from_preset(
    "resnet50_v2_imagenet", num_classes=len(class_names))

model.compile(optimizer=AdamW(learning_rate=0.001),loss='sparse_categorical_crossentropy',
              metrics = ['acc'])

model.load_weights("anim_class_model.weights.h5")

def classify_image(path):
  image = Image.open(path)

  # Convert the image to a numpy array
  image = np.array(image)

  # Normalize the image
  image = image / 255.0

  # Add a batch dimension
  image = np.expand_dims(image, axis=0)

  resizing = keras_cv.layers.Resizing(
    224, 224, crop_to_aspect_ratio=True
  )
  np_im_rs = resizing(image)

  predictions = model.predict(np_im_rs)
  return class_names[np.argmax(predictions[0])]

gr.Interface(fn=classify_image,
             inputs=gr.Image(type="filepath"),
             outputs="text").launch(debug='True') 