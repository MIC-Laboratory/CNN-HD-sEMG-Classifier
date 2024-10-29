import torch
import tensorflow as tf
import onnx
import numpy as np
import os
import binascii
from onnxsim import simplify
import numpy as np

def convert_to_c_array(bytes) -> str:
    """
    Convert TFlite model to C array
    """
    hexstr = binascii.hexlify(bytes).decode("UTF-8")
    hexstr = hexstr.upper()
    array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
    array = [array[i:i+10] for i in range(0, len(array), 10)]
    return ",\n  ".join([", ".join(e) for e in array])
tflite_binary = open('saved_model/M5_recover_float32.tflite', 'rb').read()
ascii_bytes = convert_to_c_array(tflite_binary)
header_file = "const unsigned char model_tflite[] = {\n  " + ascii_bytes + "\n};\nunsigned int model_tflite_len = " + str(len(tflite_binary)) + ";"
with open("pretrain_model/tf_lite_model/model.h", "w") as f:
    f.write(header_file)
