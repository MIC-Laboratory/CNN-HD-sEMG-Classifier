#!/bin/bash

onnx2tf -i pretrain_model/onnx_model/M5_recover.onnx -osd && \
python TF_to_C_header.py