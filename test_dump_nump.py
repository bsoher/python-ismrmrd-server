#!/usr/bin/env python

# Copyright (c) 2023-2024 Brian J Soher - All Rights Reserved
# 
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without explicit permission.


# Python modules
import os

# 3rd party modules
import numpy as np
import pydicom
import pydicom.dicomio
from pydicom.values import convert_numbers

# Our modules







#------------------------------------------------------------------------------

def main(fname=None):

    # This function is for profiling with cProfile

    fdump = r'D:\tmp\epsi\save_fire_addin_out_including_dicom\dump_nump_ch03_z00_50x51200_c64_0008.bin'
    fdicm = r'D:\tmp\epsi\save_fire_addin_out_including_dicom\Dicom_spe_00008.dcm'
    fwrit = r'D:\tmp\epsi\save_fire_addin_out_including_dicom\WriteToFile_00008.spe'


    ds = pydicom.dicomio.read_file(fdicm)
    data_shape = (ds['Rows'].value, ds['Columns'].value)
    dataf = convert_numbers(ds['SpectroscopyData'].value, True, 'f')  # (0x5600, 0x0020)
    data_iter = iter(dataf)
    data = [complex(r, i) for r, i in zip(data_iter, data_iter)]
    ddicm = np.fromiter(data, dtype=np.complex64)
    ddicm.shape = data_shape

    ddump = np.fromfile(fdump, dtype=np.complex64)
    ddump.shape = data_shape

    dwrit = np.fromfile(fwrit, dtype=np.complex64)
    dwrit.shape = data_shape

    print("ddicm and dwrit are allclose - "+str(np.allclose(ddicm, dwrit)))
    print("ddicm and ddump are allclose - "+str(np.allclose(ddicm, ddump)))
    print("ddump and dwrit are allclose - "+str(np.allclose(ddump, dwrit)))

    bob = 10

if __name__ == "__main__":

    main()    
    
