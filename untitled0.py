# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:06:48 2019

@author: sushant
"""
import tensorflow as tf
print(tf.__version__)

converter = tf.lite.TFLiteConverter.from_saved_model('C:/tmp/module_no_signatures')
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)