#!/usr/bin/env python3
# Copyright (c) 2025 aNoken

import onnx
import os
def extract_onnx_model_1_top(input_path, output_path):
   input_names = ["images"]
   output_names = ["/model/model.23/Concat_5_output_0"]
   onnx.utils.extract_model(input_path, output_path, input_names, output_names)

def extract_onnx_model_1_tail(input_path, output_path):
   input_names = [ "/model/model.23/Concat_5_output_0"]
   output_names = ["output0"]
   onnx.utils.extract_model(input_path, output_path, input_names, output_names)


def extract_onnx_model_2_top(input_path, output_path):
   input_names = ["images"]
   output_names = ["/model/model.23/Concat_output_0","/model/model.23/Concat_1_output_0","/model/model.23/Concat_2_output_0"]
   onnx.utils.extract_model(input_path, output_path, input_names, output_names)

def extract_onnx_model_2_tail(input_path, output_path):
   input_names = ["/model/model.23/Concat_output_0","/model/model.23/Concat_1_output_0","/model/model.23/Concat_2_output_0"]
   output_names = ["output0"]
   onnx.utils.extract_model(input_path, output_path, input_names, output_names)



extract_onnx_model_1_top("yolo11n_nms.onnx", "yolo11n_nms_No01_base.onnx")
extract_onnx_model_1_tail("yolo11n_nms.onnx", "yolo11n_nms_No01_decoder.onnx")

extract_onnx_model_2_top("yolo11n_nms.onnx", "yolo11n_nms_No02_base.onnx")
extract_onnx_model_2_tail("yolo11n_nms.onnx", "yolo11n_nms_No02_decoder.onnx")






