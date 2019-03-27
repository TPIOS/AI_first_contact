import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors,preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from car_detection import *

# with tf.Session() as test_a:
#     box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
#     boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
#     box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
#     scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.shape))
#     print("boxes.shape = " + str(boxes.shape))
#     print("classes.shape = " + str(classes.shape))

# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4) 
# print("iou = " + str(iou(box1, box2)))

# with tf.Session() as test_b:
#     scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
#     boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
#     classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
#     scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))

# with tf.Session() as test_b:
#     yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
#     scores, boxes, classes = yolo_eval(yolo_outputs)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")