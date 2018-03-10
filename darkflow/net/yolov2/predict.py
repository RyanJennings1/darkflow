import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

from googletrans import Translator

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


translator = Translator()

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

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

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
	#fontpil = ImageFont.truetype(os.path.join(self.FLAGS.imgdir, '../arialunicodems.ttf'), 30)
	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[0], thick)
			#colors[max_indx], thick)
			#'white', thick)
		if self.FLAGS.language:
			translated_mess = translator.translate(mess, dest=self.FLAGS.language)
		else:
			translated_mess = translator.translate(mess, dest='en')
		cv2.putText(imgcv, translated_mess.text + ' ' + str(float('%.2f' % confidence)), (left, top - 12),
                        0, 1e-3 * h * 3, colors[0],thick)
			#0, 1e-3 * h * 3, colors[max_indx],thick)
		fontpil = ImageFont.truetype(os.path.join(self.FLAGS.imgdir, '../arialunicodems.ttf'), int(h * 0.05))

		# thick outline
		drawpil.text((left-1, top-1), str(translated_mess.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)
		drawpil.text((left+1, top-1), str(translated_mess.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)
		drawpil.text((left-1, top+1), str(translated_mess.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)
		drawpil.text((left+1, top+1), str(translated_mess.text) + ' ' + str(float('%.2f' % confidence)), (0, 0, 0), font=fontpil)

		drawpil.text((left, top), str(translated_mess.text) + ' ' + str(float('%.2f' % confidence)), fill=tuple(map(int, colors[9])), font=fontpil)
		drawpil.rectangle(((left, top), (right, bot)), fill=None, outline=tuple(map(int, colors[9])))
		#outpil = os.path.join(self.FLAGS.imgdir, 'out')
		imgpil.save(os.path.join(outpil, os.path.basename(im)))

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	#cv2.imwrite(img_name, imgcv)
