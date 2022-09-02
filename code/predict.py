# %% [markdown]
# The ONNX runtime doesn't train models, only generates predictions.
# But it's very lightweight and fast at this!
# So if we just want ready predictions out of pre-trained EmoNet, 
# we don't have to change the ONNX model into any other formats.

# %%
# import shit

# import torch (NOT YET...)
import numpy as np
import pandas as pd
from ast import literal_eval
import onnx
import onnxruntime as ort
from pycocotools.coco import COCO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from image_utils import *

emonet_path = '../ignore/models/EmoNet.onnx'
# this only works on the server rn bc the NSD stuff is only saved there
nsd_path = '/data/eccolab/Code/NSD/'
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
# Get the COCO annotations, but only included in NSD

# this one takes a hot second to load into memory
coco = COCO(nsd_path + 'annotations/instances_train2017.json')
nsd_stims = pd.read_csv(nsd_path + 'nsd_stim_info_merged.csv',
                        converters = {'cropBox': literal_eval})
nsd_stims_1000 = nsd_stims.query('shared1000')

# %%
# read NSD shared 1000 imgs in from COCO

# some of the NSD stims outside of shared1000 don't appear in this set of COCO
# so this only works for the shared1000 subset rn
# further, UGH, coco.loadImgs _expects_ a list so it fails for single values
# thus, may as well pass the whole column into loadImgs once and list-comp from that
# instead of trying to be nice and vectorize
nsd_metadata_1000 = coco.loadImgs(nsd_stims_1000['cocoId'])
nsd_stims_1000 = nsd_stims_1000.assign(cocoUrl = [img['coco_url'] for img in nsd_metadata_1000],
                                       cocoHeight = [img['height'] for img in nsd_metadata_1000],
                                       cocoWidth = [img['width'] for img in nsd_metadata_1000])

# %%
# process NSD stimuli into arrays for EmoNet
nsd_inputs = []

for i in range(len(nsd_stims_1000)):
    path = nsd_stims_1000.cocoUrl.iloc[i]
    cropbox_nsd = nsd_stims_1000.cropBox.iloc[i]
    dims = (nsd_stims_1000.cocoWidth.iloc[i], nsd_stims_1000.cocoHeight.iloc[i])
    cropbox_pillow = cropbox_nsd_to_pillow(dims, cropbox_nsd)

    img = read_and_preprocess_image(path, cropbox = cropbox_pillow)
    # onnxruntime expects a list of inputs
    # so even with one input, ye must list it
    nsd_inputs.append(img)

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
                           input_feed = {input_name: nsd_inputs})[0]
preds = [emonet_output_classes[pred] for pred in np.argmax(probs, axis = 1)]
# %%
# output EmoNet probs as dataframe for saving to CSV

# make the class labels into variable-safe pothole case
probs_df = pd.DataFrame(probs, columns=[emo.lower().replace(' ', '_') for emo in emonet_output_classes])
# the df indices don't line up so force-coerce the coco_id column to list before binding on
probs_df = probs_df.assign(coco_id = nsd_stims_1000.cocoId.tolist())
probs_df = probs_df.set_index('coco_id')
# THIS EXPORTS! CREATES FILE!
probs_df.to_csv('../ignore/outputs/emonet_probs_nsd_shared1000.csv')
# %%
# t-SNEEEE
tsne = TSNE(n_components=2, perplexity = 50, learning_rate='auto')
nsd_emonet_embedding = tsne.fit_transform(probs)
nsd_emonet_embedding = pd.DataFrame(nsd_emonet_embedding, columns = ['x', 'y'])
nsd_emonet_embedding = nsd_emonet_embedding.assign(emo_pred = preds)
# %%
sns.scatterplot(
    x = 'x',
    y = 'y',
    hue = 'emo_pred',
    data = nsd_emonet_embedding
)
# %%
