import numpy as np
import os
file_name = ["out.log", "expect.log"]
batch_dir = "../batch_reports/"
def log(data, file_idx):
        f_out = open(file_name[file_idx], "a")
        f_out.write(data)
        f_out.close()

def batchLog(data, epoch, batchType):
        f_name = batch_dir + epoch + "_" + batchType
        f_out = open(f_name, "w")
        f_out.write(data)
        f_out.close
        
def fileSetup():
    for f in file_name: 
        f_out = open(f, "w")
        f_out.close()