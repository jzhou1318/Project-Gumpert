# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import wikipedia as wiki
import tensorflow as tf
import cv2 as cv
import re
import os
import numpy as np
import unicodedata
import urllib

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def getUrl(test):
    for urls in test.images:
        names = urls.split('/')
        name = names[-1]
        names = name.split('.')
        name = names[0]
        ext = names[1]
        name = name.replace('_', ' ')
        title = imgx.title.replace('-', ' ')
        if ext == 'jpg':
            url.append(urls)
        # if (name.find(remove_accents(title)) != -1) and ext == 'jpg':
        #     print(urls)
        #     url.append(urls)
        #print(urls)
    sorted_list = sorted(url, key=len)
    #print(sorted_list)
    #imgUrl = min(url, key=len)
    imgUrl = sorted_list[0]
    resp = urllib.request.urlopen(imgUrl)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    factor = (500/img.shape[0])
    img =cv.resize(img,(0,0),500,factor,factor)
    cv.imshow(title, img)
    cv.waitKey()
    cv.destroyAllWindows()

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  

  filew = open("testfile.txt","w")
  file= open('/Accounts/zhouj2/Documents/landmarkNames.txt')
  listLand=[]
  for line in file:
    listLand.append(line)
 
  for i in top_k:
    #print("COUNT")
    #print(labels[i], results[i])
    it=labels[i].strip()
    for land in listLand:
      #print("COUNT2")
      output=land.lower().strip()
      if output==it:
        wikA = re.findall('[A-Z][^A-Z]*', land)
        wik=""
        for word in wikA:
          wik+=word+" "
        print(wik,results[i])
        filew.write("\n")
        filew.write(wik)
        filew.write(str(results[i]))
        filew.write("\n")
        if results[i]>0.3:
          try:
            wEntry=wiki.summary(wik, sentences = 2).replace("( listen)","").replace(" locally also ", '').replace(" ( ,)", '')
            print(wEntry)
            filew.write(wEntry)
            filew.write("\n")
            url = []
            try:
              imgx = wiki.page(wik)
 #             getUrl(imgx)
            except wiki.exceptions.DisambiguationError as e:
              imgx = wiki.page(e.options[0])

            except:
              print("fuck")
          except:
            print("WIKI PAGE NOT FOUND")
            filew.write("WIKI PAGE NOT FOUND")
            filew.write("\n")
  filew.close()
  os.system("open "+"testfile.txt")
  if imgx is not None:
      getUrl(imgx)
    
        
##      arr = wEntry.split(" ")
##      P=""
##      Pf=""
##      for word in arr:
##        word1=word.lower()
##        if word1 in labels[i] and word1 not in Pf and len(word1)>3:
##          Pf+=word1
##          P+=word+" "
##      if len(P)!=0:
##        print(P,str(results[i]))
##      else:
##        print(labels[i],results[i])
##      if (results[i] > 0.30):
##        try:
##          print(wEntry)
##        except:
##          print("WIKI PAGE NOT FOUND")
          
 



