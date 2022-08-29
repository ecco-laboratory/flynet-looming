# %%
# import shit

# import torch (NOT YET...)
import numpy as np
import onnx
import onnxruntime as ort
from image_utils import read_and_preprocess_image

emonet_path = '../ignore/models/EmoNet.onnx'

# %% 
# A tensor is like an array, but safe for consumption by modeling thingies
emonet_onnx = onnx.load(emonet_path)
# %%
emonet_session = ort.InferenceSession(emonet_path,
                                      providers = ort.get_available_providers())
input_name = emonet_session.get_inputs()[0].name
# %%
# Get and preprocess those dank dank imgs
paths = [
    '/Users/mthieu/Downloads/IMG_0033.jpeg',
    'https://pbs.twimg.com/media/C3sLsPnUEAEcTrZ.jpg',
    'https://cdn.cnn.com/cnnnext/dam/assets/191019180832-nyc-cigarette-cockroach-viral-video-trnd.jpg',
    '/Users/mthieu/Downloads/conch_eyes.png',
    '/Users/mthieu/Downloads/IMG_4820.jpg'
    ]
inputs = []

for path in paths:
    img = read_and_preprocess_image(path)
    # onnxruntime expects a list of inputs
    # so even with one input, ye must list it
    inputs.append(img)
# %%
# There seems no better way... defining EmoNet's output classes
emonet_output_classes = [
    'Adoration',
    'Aesthetic Appreciation',
    'Amusement',
    'Anxiety',
    'Awe',
    'Boredom',
    'Confusion',
    'Craving',
    'Disgust',
    'Empathic Pain',
    'Entrancement',
    'Excitement',
    'Fear',
    'Horror',
    'Interest',
    'Joy',
    'Romance',
    'Sadness',
    'Sexual Desire',
    'Surprise'
]
# %%
# Predictions!?
probs = emonet_session.run(output_names=['softmax'],
                           input_feed = {input_name: inputs})[0]
preds = [emonet_output_classes[pred] for pred in np.argmax(probs, axis = 1)]
# %%
