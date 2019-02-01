import json
import math
import os

#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

import cv2
from googletrans import Translator
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def expit(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
    # meta
    meta = self.meta
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

def postprocess(self, net_out, im, save=True):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = self.findboxes(net_out)

    confidence_threshold = 0.5

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape

    outpil = os.path.join(self.FLAGS.imgdir, 'out')
    imgpil = Image.open(os.path.join(self.FLAGS.imgdir, os.path.basename(im)))
    drawpil = ImageDraw.Draw(imgpil)

    resultsForJSON = []

    # Boxes is the list of prediction boxes in the image
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, label, max_indx, confidence = boxResults

        if confidence > confidence_threshold:
            thick = int((h + w) // 300)
            if self.FLAGS.json:
                resultsForJSON.append({"label": label, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
                continue

            predicted_text = label

            if self.FLAGS.language:
                translator = Translator()
                predicted_text = translator.translate(label, dest=self.FLAGS.language)

            fontpil = ImageFont.truetype(os.path.join(self.FLAGS.imgdir, '../arialunicodems.ttf'), int(h * 0.05))

            # Draw a thick outline of the text
            drawpil.text((left-1, top-1),
                         str(predicted_text.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)
            drawpil.text((left+1, top-1),
                         str(predicted_text.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)
            drawpil.text((left-1, top+1),
                         str(predicted_text.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)
            drawpil.text((left+1, top+1),
                         str(predicted_text.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)

            drawpil.text((left, top), str(predicted_text.text) + ' ' + str(float('%.2f' % confidence)), fill=tuple(map(int, colors[9])), font=fontpil)
            drawpil.rectangle(((left, top), (right, bot)), fill=None, outline=tuple(map(int, colors[9])))

        if not save: return imgcv

        outfolder = os.path.join(self.FLAGS.imgdir, 'out')
        img_name = os.path.join(outfolder, os.path.basename(im))
        if self.FLAGS.json:
            textJSON = json.dumps(resultsForJSON)
            textFile = os.path.splitext(img_name)[0] + ".json"
            with open(textFile, 'w') as f:
                f.write(textJSON)
            return

        imgpil.save(os.path.join(outpil, os.path.basename(im)))