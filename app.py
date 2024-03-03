import gradio as gr
from fastai.vision.all import *
import pathlib

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('model.pkl')

categories = ('Magic the Gathering', 'Pokemon',  'Yugioh')
def classify_image(img):
    img = PILImage.create(img)
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

title = 'Card Classifier'
description = 'Classifies cards from Magic the Gathering, Pokemon, and Yugioh.'
examples = ['mtg.jpg', 'pokemon.jpg', 'yugioh.png', 'mtg_hard.jpg']
image = gr.components.Image(width=192, height=192)
label = gr.components.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, title=title, description=description)
intf.launch()
