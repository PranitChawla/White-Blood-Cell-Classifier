import tensorflow as tf
from lxml import etree
from object_detection.utils import dataset_util
import io
import PIL.Image
import hashlib
flags = tf.app.flags
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(data,label_dict):
  # TODO(user): Populate the following variables from your example.

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  height = data['size']['height'] # Image height
  height=int(height)
  width = data['size']['width'] # Image width
  width=int(width)
  filename = data['filename'] # Filename of the image. Empty if image is not from file
  #encoded_image_data = None # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'
  img_path="/home/pranit/Desktop/white_blood_cell_project/BCCD_Dataset/BCCD/JPEGImages/"+data['filename']
  truncated = []
  poses = []
  difficult_obj = []
  with tf.gfile.GFile(img_path,'rb') as fid:
    encoded_jpg=fid.read()
  encoded_jpg_io=io.BytesIO(encoded_jpg)
  image=PIL.Image.open(encoded_jpg_io)
  if image.format!='JPEG':
    raise ValueError('Image format not JPEG')
  key=hashlib.sha256(encoded_jpg).hexdigest()
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if difficult:
      continue
    difficult_obj.append(int(difficult))
    xmins.append(float(obj['bndbox']['xmin']) / float(width))
    ymins.append(float(obj['bndbox']['ymin']) / float(height))
    xmaxs.append(float(obj['bndbox']['xmax']) / float(width))
    ymaxs.append(float(obj['bndbox']['ymax']) / float(height))
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))
    # print ("length of array is ",len(xmins))
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/key/sha256' : dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return tf_example


def main(sets,label_dict):
  # label_path="/home/pranit/Desktop/white_blood_cell_project/labels.pbtxt"
  # label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  output_path="/home/pranit/Desktop/white_blood_cell_project"
  output_path=output_path+"/"+sets+".record"
  writer = tf.python_io.TFRecordWriter(output_path)
  path="/home/pranit/Desktop/white_blood_cell_project/BCCD_Dataset/BCCD/ImageSets/Main/"
  xml_path="/home/pranit/Desktop/white_blood_cell_project/BCCD_Dataset/BCCD/Annotations/"
  path=path+sets+".txt"
  f=open(path,"r")
  line=f.read()
  list_of_imgs=line.split('\n')
  print ("length of image array is ",len(list_of_imgs))
  cnt=0
  for image in list_of_imgs:
    # print (image)
    if (len(image)==0):
      continue
    image=image+".xml"
    xml_paths=xml_path+image
    with tf.gfile.GFile(xml_paths, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    tf_example = create_tf_example(data,label_dict)
    # print ("writing example",cnt+1,"of ",len(list_of_imgs))
    cnt+=1
    writer.write(tf_example.SerializeToString())

  writer.close()
