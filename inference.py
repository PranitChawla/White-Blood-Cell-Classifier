#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[15]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util
from utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
	"""loads gdal image as a numpy array
	
	Arguments:
		image {tif} -- image in .tif format
	
	Returns:
		np array -- image in np array
	"""
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



def run_inference_for_single_image(image, graph):
	"""forms the output_dict for a single image by passing the image to the model for detection
	
	Arguments:
		image {tif} -- image
		graph {frozen_inference_graph.pb} -- inference graph of model 
	
	Returns:
		dictionary -- dictionary containing bounding boxes and confidence scores
	"""
	with graph.as_default():
		with tf.Session() as sess:
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [ 'num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
			if 'detection_masks' in tensor_dict:
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
				detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict


def create_inference_graph(PATH_TO_FROZEN_GRAPH):
	"""forms inference graph from frozen graph directory
	
	Arguments:
		PATH_TO_FROZEN_GRAPH {path} -- path to the frozen graph directory
	
	Returns:
		 graph -- inference graph of model
	"""
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	return detection_graph





def create_bounding_boxes_from_tifs(output_dict,image):
	"""get the ouput bounding boxes and scores for each tiff 
	
	Arguments:
		detection_graph {graph} -- inference graph of model
		output_tiles_path {path} -- path to output folder contatining smaller tifs
		tif {tif} -- smaller image in .tif format
		img {tif} -- bigger image in .tif format
		content {list} -- list of output_tiles_path directory 
	
	Returns:
		lists -- lists of bounding boxes,scores
	"""
	boxes = np.squeeze(output_dict['detection_boxes'])
	scores = np.squeeze(output_dict['detection_scores'])
	classes=np.squeeze(output_dict['detection_classes'])
	min_score_thresh = 0.7
	bboxes=[boxes[x] for x in range(len(scores)) if scores[x]>min_score_thresh and classes[x]==2]
	sscores=[scores[x] for x in range (len(scores)) if scores[x]>min_score_thresh and classes[x]==2]
	cclasses=[classes[x] for x in range (len(classes)) if scores[x]>min_score_thresh and classes[x]==2]
	im_width, im_height = image.size
	final_box = []
	for box in bboxes:
		ymin, xmin, ymax, xmax = box
		final_box.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])
	return bboxes,sscores,cclasses










def create_results(PATH_TO_FROZEN_GRAPH, PATH_TO_TEST_IMAGES_DIR,PATH_TO_LABELS):
	"""function which calls all functions to create object detection results
	
	Arguments:
		PATH_TO_FROZEN_GRAPH {path} -- path to frozen graph 
		PATH_TO_TEST_IMAGES_DIR {path} -- path to input tiffs
		OUTPUT_SHP_PATH {path} -- path to folder containing all shape files
		HEIGHT {int} -- ysize of smaller tif
		WIDTH {int} -- xsize of smaller tif
	"""
	detection_graph=create_inference_graph(PATH_TO_FROZEN_GRAPH)
	content=os.listdir(PATH_TO_TEST_IMAGES_DIR)
	for img in content:
		image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,img)	
		image = Image.open(image_path)
		image_np = load_image_into_numpy_array(image)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
		final_boxes,scores,classes=create_bounding_boxes_from_tifs(output_dict,image)
		print (scores)
		final_boxes=np.array(final_boxes)
		scores=np.array(scores)
		classes=np.array(classes)
		category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
		vis_util.visualize_boxes_and_labels_on_image_array(image_np,final_boxes,classes,scores,category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)
		# cv2.imshow("image",image_np)
		img1 = Image.fromarray(image_np)
		save_path="/home/pranit/Desktop/white_blood_cell_project/test_results"
		# img1.show()
		# cv2.waitKey(0)
		img1.save(save_path+"/"+img)
	      
	      
	      
	      
	      
	      