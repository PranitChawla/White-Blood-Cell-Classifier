from lxml import etree
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
label_map_path="/home/pranit/Desktop/white_blood_cell_project/labels.pbtxt"
# label_map_dict = label_map_util.get_label_map_dict(label_map_path)
# print (label_map_dict)
path="/home/pranit/Desktop/white_blood_cell_project/BCCD_Dataset/BCCD/Annotations/BloodImage_00000.xml"
with tf.gfile.GFile(path, 'r') as fid:
	xml_str = fid.read()
xml = etree.fromstring(xml_str)
data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
for i in data['object']:
	print (i['name'])
	print (i['bndbox'])
# path="/home/pranit/Desktop/white_blood_cell_project/BCCD_Dataset/BCCD/ImageSets/Main/train.txt"
# f=open(label_map_path,"r")
# line=f.read()
# # list_of_imgs=line.split('\n')
# # for image in list_of_imgs:
# print (line)
