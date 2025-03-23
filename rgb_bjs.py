
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    imgGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)):
                    acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, config, metadata)
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):
                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()
                    connection.send_image(item)
                    continue

            # Images and waveform data are not supported in this example
            elif isinstance(item, ismrmrd.Acquisition) or isinstance(item, ismrmrd.Waveform):
                continue

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, config, metadata)
            connection.send_image(image)
            imgGroup = []

    finally:
        connection.send_close()

def process_raw(group, config, metadata):
    if len(group) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha PE RO phs] array
    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    phs = [acquisition.idx.phase                for acquisition in group]

    # Use the zero-padded matrix size
    data = np.zeros((group[0].data.shape[0], 
                     metadata.encoding[0].encodedSpace.matrixSize.y, 
                     metadata.encoding[0].encodedSpace.matrixSize.x, 
                     max(phs)+1), 
                    group[0].data.dtype)

    rawHead = [None]*(max(phs)+1)

    for acq, lin, phs in zip(group, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            # TODO: Account for asymmetric echo in a better way
            data[:,lin,-acq.data.shape[1]:,phs] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead[phs] is None) or (np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])):
                rawHead[phs] = acq.getHead()

    # Flip matrix in RO/PE to be consistent with ICE
    data = np.flip(data, (1, 2))

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    data *= np.prod(data.shape) # FFT scaling for consistency with ICE

    # Sum of squares coil combination
    # Data will be [PE RO phs]
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    # Normalize and convert to int16
    data *= maxVal/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.x)/2)
    data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.x]

    # Remove phase oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.y)/2)
    data = data[offset:offset+metadata.encoding[0].reconSpace.matrixSize.y,:]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Format as ISMRMRD image data
    imagesOut = []
    for phs in range(data.shape[2]):
        # Create new MRD instance for the processed image
        # data has shape [PE RO phs], i.e. [y x].
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        tmpImg = ismrmrd.Image.from_array(data[...,phs], transpose=False)

        # Set the header information
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[phs]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = phs

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = str((maxVal+1)/2)
        tmpMeta['WindowWidth']            = str((maxVal+1))
        tmpMeta['Keep_image_geometry']    = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    # Call process_image() to create RGB images
    imagesOut = process_image(imagesOut, config, metadata)

    return imagesOut

def process_image(images, config, metadata):
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    if data.shape[3] != 1:
        logging.error("Multi-channel data is not supported")
        return []
    
    # Normalize to (0.0, 1.0) as expected by get_cmap()
    data = data.astype(float)
    data -= data.min()
    data *= 1/data.max()

    # Apply colormap
    cmap = plt.get_cmap('jet')
    rgb = cmap(data)

    # Remove alpha channel
    # Resulting shape is [row col z rgb img]
    rgb = rgb[...,0:-1]
    rgb = rgb.transpose((0, 1, 2, 5, 4, 3))
    rgb = np.squeeze(rgb, 5)

    # MRD RGB images must be uint16 in range (0, 255)
    rgb *= 255
    data = rgb.astype(np.uint16)
    np.save(debugFolder + "/" + "imgRGB.npy", data)

    data[0:9,:,:,0,:] = 128
    data[0:9,:,:,1,:] = 0
    data[0:9,:,:,2,:] = 0
    data[:,0:9,:,0,:] = 0
    data[:,0:9,:,1,:] = 128
    data[:,0:9,:,2,:] = 0

    currentSeries = 0

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):
        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)
        data_type = imagesOut[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type

        # Set RGB parameters
        oldHeader.image_type = 6  # To be defined as ismrmrd.IMTYPE_RGB
        oldHeader.channels   = 3  # RGB "channels".  This is set by from_array, but need to be explicit as we're copying the old header instead

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'RGB']
        tmpMeta['SequenceDescriptionAdditional']  = 'FIRE_RGB'
        tmpMeta['Keep_image_geometry']            = 1

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml


    if True:
        
        dat = []
        for data in [bdat1, bdat2, bdat2b, bdat2c, bxvals]:
            data = np.load(io.BytesIO(zlib.decompress(base64.b64decode(data))))
            dat.append(data)

        rgb2 = build_fig(dat, figsize=(11, 8.5), dpi=100)

        # Remove alpha channel
        # Resulting shape is [row col z rgb img]
        rgb2 = rgb2[...,0:-1]
        rgb2 = rgb2.transpose((0, 1, 2, 5, 4, 3))
        rgb2 = np.squeeze(rgb2, 5)

        xtraData = rgb2.astype(np.uint16)
        imgX, imgY, _, _, _ = rgb2.shape

        # Create new MRD instance for the xtra image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        xtraImg = ismrmrd.Image.from_array(xtraData[...,0].transpose((3, 2, 0, 1)), transpose=False)
        data_type = xtraImg.data_type

        # Set the header information
        # Set RGB parameters

        tmpHead = head[0]
        tmpHead.data_type = data_type
        
        tmpHead.image_type = 6  # To be defined as ismrmrd.IMTYPE_RGB
        tmpHead.channels   = 3  # RGB "channels".  This is set by from_array, but need to be explicit as we're copying the old header instead
        tmpHead.field_of_view = (ctypes.c_float( imgX), ctypes.c_float( imgY), ctypes.c_float(10))  # Dummy FOV because the spectroscopy FOV isn't appropriate
        tmpHead.matrix_size   = (ctypes.c_ushort(imgX), ctypes.c_ushort(imgY), ctypes.c_ushort(1))
        
        xtraImg.setHead(tmpHead)

        tmpMeta = meta[0]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'RGB_BJS']
        tmpMeta['SequenceDescriptionAdditional']  = 'FIRE_RGB_XTRA'
        tmpMeta['Keep_image_geometry']            = 1
        tmpMeta['InternalSend']                   = ['bool', 'true']

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(tmpHead.read_dir[0]), "{:.18f}".format(tmpHead.read_dir[1]), "{:.18f}".format(tmpHead.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(tmpHead.phase_dir[0]), "{:.18f}".format(tmpHead.phase_dir[1]), "{:.18f}".format(tmpHead.phase_dir[2])]

        xtraImg.attribute_string   = tmpMeta.serialize()      
        xtraImg.slice = 3  
        
        imagesOut.append(xtraImg)

    if True:
        
        dat = []
        for data in [bdat1, bdat2, bdat2b, bdat2c, bxvals]:
            data = np.load(io.BytesIO(zlib.decompress(base64.b64decode(data))))
            dat.append(data)

        rgb3 = build_fig(dat, figsize=(11, 8.5), dpi=150)

        # Remove alpha channel
        # Resulting shape is [row col z rgb img]
        rgb3 = rgb3[...,0:-1]
        rgb3 = rgb3.transpose((0, 1, 2, 5, 4, 3))
        rgb3 = np.squeeze(rgb3, 5)

        xtraData3 = rgb3.astype(np.uint16)
        imgX, imgY, _, _, _ = rgb3.shape

        # Create new MRD instance for the xtra image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        xtraImg3 = ismrmrd.Image.from_array(xtraData3[...,0].transpose((3, 2, 0, 1)), transpose=False)
        data_type = xtraImg3.data_type

        # Set the header information
        # Set RGB parameters

        tmpHead3 = head[1]
        tmpHead3.data_type = data_type
        
        tmpHead3.image_type = 6  # To be defined as ismrmrd.IMTYPE_RGB
        tmpHead3.channels   = 3  # RGB "channels".  This is set by from_array, but need to be explicit as we're copying the old header instead
        tmpHead3.field_of_view = (ctypes.c_float( imgX), ctypes.c_float( imgY), ctypes.c_float(10))  # Dummy FOV because the spectroscopy FOV isn't appropriate
        tmpHead3.matrix_size   = (ctypes.c_ushort(imgX), ctypes.c_ushort(imgY), ctypes.c_ushort(1))
        
        xtraImg3.setHead(tmpHead3)

        tmpMeta3 = meta[1]
        tmpMeta3['DataRole']                       = 'Image'
        tmpMeta3['ImageProcessingHistory']         = ['PYTHON', 'RGB_BJS']
        tmpMeta3['SequenceDescriptionAdditional']  = 'FIRE_RGB_XTRA'
        tmpMeta3['Keep_image_geometry']            = 1
        tmpMeta3['InternalSend']                   = ['bool', 'true']

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta3.get('ImageRowDir') is None:
            tmpMeta3['ImageRowDir'] = ["{:.18f}".format(tmpHead3.read_dir[0]), "{:.18f}".format(tmpHead3.read_dir[1]), "{:.18f}".format(tmpHead3.read_dir[2])]

        if tmpMeta3.get('ImageColumnDir') is None:
            tmpMeta3['ImageColumnDir'] = ["{:.18f}".format(tmpHead3.phase_dir[0]), "{:.18f}".format(tmpHead3.phase_dir[1]), "{:.18f}".format(tmpHead3.phase_dir[2])]

        xtraImg3.attribute_string   = tmpMeta3.serialize()      
        xtraImg3.slice = 3  
        
        imagesOut.append(xtraImg3)


    return imagesOut



import numpy as np
import io
import zlib
import base64
import datetime
import time

from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg


NUMERIC_LIST_ENCODING = "npy zlib base64"

fit_data = [['      Metab Results', '     Area    ', '   CrRao[%]  ', '  CnfInt[%]   '],
            ['  choline-truncated', '    11.397   ', '    101.54   ', '      50      '], 
            ['           creatine', '    21.749   ', '    1.1837   ', '    5.3597    '], 
            ['          glutamate', '    14.642   ', '     4.26    ', '    18.065    '], 
            ['       myo-inositol', '    22.793   ', '    1.8381   ', '    52.677    '], 
            ['  n-acetylaspartate', '    32.57    ', '    1.3128   ', '    3.9967    '], 
            ['    scyllo-inositol', '    3.0723   ', '    188.8    ', '    2.8798    '], 
            ['            taurine', '    7.5977   ', '    7.5569   ', '    55.462    '], 
            ['     Global Results', '    Value    ', ' CrRao[delta]', 'CnfInt[delta] '], 
            ['                 Ta', '   0.13693   ', '  0.0082428  ', '      0       '], 
            ['                 Tb', '   0.14416   ', '  0.0054326  ', '      0       '], 
            ['             Phase0', ' 0.0 (-212.1)', '   0.36204   ', '      0       '], 
            ['             Phase1', ' 0.0 (-261.8)', '    8.9632   ', '      0       '], 
            ['       Calc Results', '     Value   ', '             ', '              '], 
            ['          Linewidth', '    5.126    ', '             ', '              '], 
            ['          ChiSquare', '   0.012599  ', '             ', '              '], 
            [' Weighted ChiSquare', '  0.0035652  ', '             ', '              '], 
            ['  Math Finite Error', '    False    ', '             ', '              ']]

bdat1 = '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'
bdat2 = 'eNqdyPs3FgYcx3Edt0ZzySXm2jrSYyiXLLfv54siKztWdDmKHjxmOIhSSYVctj1EhUpyrZQolVSuRxGJJz2dbtLa6HIiClknGmv/wt6/vV+Hf/T18t44SyZOZrdpsCg2KMbUwdjUKcTW1NzYNCQqZluMMDIgKiZY9J+7CyNiRV88NlQYLfryi2ysrcwF5sZ7jP93SssKLtHBWjn0mgxS7/FuOjn6hHo72kl4fy7KdinAc/0IXVI3gnXGSjz0L2yWb9sOqX4alvfpkff5n5qCu/3QFuiH+H3CpkXfnSWNwGV4b22F1M7ZKA8rprCpl+T+TSX57PCnty4aFO62naYzGpyjUjop0doEOv5O2GdnhPorAjSkacIzlDDfwRZzH7nCJsML2/yi8FiwFImyWsjq34WYcE2kKqvjXIobTuSYYd3VANSXu+HeV0I8p0RUtuTCzbgSydEvcHfHLA4TK/GFhTosEqty9toJ1CoMwWbnFEoXanBFoA/HLYjnFv0sflSUx+reOTxbKYUXDwdz48eVrCoWsHy/CavaCfj4A30eNtLhje9M+NMNQ756TIdfvVHi9R5arJZuzjNKAtbKdWQNY0vuX2HLEXtc+a10HSfVRrFepZiFqjncFp7Nxw1SWaTmxdYTqnxd0AepsAo9R6oRLF+P1PbHiB9vgbDtLhzvXELHztuIllaj/0gxBlf/BiP7SLysSkar8xkMeMuzNGI1p/aJWT2zhCVBRbz4QAJvLgXb5ypx+mUJWovysWTlVljRGRj6NiL05gLOdtjNy0Zy+fXRfLY8nc5B7T4s+aDJlcsvY/+kF56LLTE5ZI09z1Zhc3UTJbR/pMN9WtAM14bETAENuQ5Y/IcFLl43RFePC6Y3zcf0FQ3Eja7AAfMtkLdPQ+Dtvcjyb8T5zls4t/sYHvrG4VDtCpR0CrC6dJSOpnpC+VcHdCVZoEuqj4jqJTj0JhaxD05BN7AbVeduYmiwFnFZLWiVlMDz+zKUdlXg8OgN6CX/BZWEV7CvkmHbUnm2uzCBxYayHLJShu98VuWDj2R5ofskbAoVWe0HGR7rkmHHKUW+a6zLa2JUeAOG4aunxPl5FpyQkcRin9s8oyjrMqOr4FKa/ZkDJhu49WAC5/1jziP31Xj9tALH/t6L1+J8SDxLsFb1FOQCQtBqkIjQmsPQ1XaE4qAMehyVkSmnDUHzJzIq/5O8tcZpzTUJdY11ERUrI/QiQ6qtgdJmDZTXOUD8IRH6SeUoH7mB5PQmNIs6MOBWhsC1eRjfko59Y35Q3OGLdNFSvKxwRXeoCoJmjGDZeovCTjc6Z6YH0KyhVAo42+gslP7d4BSp7Zxxco3TPIMlTdI7PY33Basbh2I0KXOgjPRFxRQdWkPvztRTwWUJPTYLp07LAsLeQlp1NIGePXxIua0Sqnj6jp7I9tGmQgPsPeUB28xk3LeKgHNRAS7XiRHmXo2nOQ2IvVKHHP9bSAnrwLzUW0j85SlqOq8hrbIQWuPZyJLbhlVzYlAQ747A4XnQafOEh+5ObL3nBDNFFXxeogbTATt0Slyx4Z0VavzlIBprJv8nSgh+b4yZMgtEVE2Q9WZZDKi10PXBUuemn11o4/Jw8ihOpJBiMTlZHKHIAw+oyS+WajSnnFS+nUs+x0yo+IUezbq00fmEmQY96zSiMxWRJKPcSPOKVJCUORv79erovUE+zTn5hmy+bqcTq5bSvwtSP/4='
bdat2b = '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'
bdat2c = 'eNqdk/k3lQkAhskoKQxps0wSiahTxmjxvS/aNXWairSobiIZrq3JKUnR1T32LCENiqJoLpdk+0q7pRkpVFJTZ9KumYyl2RjzL8zz2/uc99cnZbXrqjUbVVX2qoRZ7PQK8Qy2WGBi4eD9pYW1iYV3QPCeYIl0W0DwTq///FKJX4jXsA/xkQR6De8ZtnPnWFtam4Sb/G80hbYKbFtVidEDlfDXq4JdSyU2RVbCXbMSuWsuodWzAg24CL9x5XgzrQzNfkos6CnFYH8pxmUpEcRyHBv+bQquwV6POpRcuwntmU3oenIXtroPUN76BAaSLtgs7obkTS9avP7Gg8eqjLVUp9bQKH7RM5pSuSbnz9Nkj89odvlo8FLZKJaZjqJj0kgqjEaysV6d4Y/U6V4xkq6OGny9WZNu6Vp0dtJj2uAkqmaZcor7LCYFL+TC4hW0P+JOr/u7uCorlOp5UZwyNZaK7cmcJs2kYkMuBRSw5eMF/ra4nFWW1XQ9coVRW27wWUM9bxbcoff9Zhrsu8ecC60MPd1OnZyHPO/SwR/5mLrhj2nztoNY9ohph9rZO/UewwqbOJhwnWuXVrNDv4SGgWdYv/wEFx1N4D/ekfQw8mPBknV827iA4fGm/NVTk9obX0LeW418SSpCSnfDKsIOQ3c0oNF+U2iviBLuf9QX7Mw3OGSbD9ZoBr6tHS1bIm7Njhbr2vLFthvV4uxrjeLZCQ/ESXOeibVOr8Q4xTtxsfy9+ER4KxaodIm7v+0UNfzvivs860SPo0XioCJZ1JH6iytrZ4tqkR21wfkGtdvXajuMN3ju0GVJ4fOKo8JnqqWCZFeHYO2rioNXxkMr1AqrHR2xwsQd2UWBmD31CJJ9EqH9eyZMHfLwwrgIOzJLcOu4EsaTlLDuUeDnD8V473we9/wKYKaSj71GuXgqz0KjRTqWz03FX9JkNF89hoZjKVDLyIDNudNo7S/D2JhGeB94gZwQFVrp6XC+6RdUNtmwK1ygTv3X7EzdzBNZu1mnHcqtxpEMspIza1c87cOPMa4/hX/OSqN+YhpbDNJYOJDCzDXJ/G5WEgd643lXL46L5sUww1DOtYUyOp+LpL7bQXbk7qVfoz+fhnnysqk7+z+tZGg22fXclkvtZ1C+zIjpt3XoZDyCf+S8xkvHZhRpVGKEWx4cZ6agdLMcyydGQzqUiIZ7p9F8pgaOUQ9RZt0PoxpNqpw15H0nG9q7kbesv+GesxLqjgui+rUIuo6MZmd4DLPc4pjqE8PIUhm1PA7SMUJKcxcP3m52YWbzV9RJMOX0aTo81P4JV+zakHRSie6qRDis2Iox2eaQ+fYIk7+vFdqbTgmXZRlC1WCu0CerFJ72dAgNp9RwZ5kB9u+wRt9VAVq6a2Fm7Asj/SPom3wccYnnEdp5DUbrn2HgqgrLFXpcN9+cEzfZc7d8BTcabOFRvQCmbT9Ms+GGku1T6a84SZfIfA4aFfGKbQkThpSMT1OypUrBTqsi/rLwDKfEZzNsUjpjaxPo0S2jXG8/Tab7sHiHGw2jFnPCxLlMsTHhnlgdXtVVpdmjd1gS2QHb+Xcw86c6KL0uQTtegb715xC4MA+txVnoLElBjWkcXvnKMM3rEJyc9+H25BD80C1FwJAf+j/4IadWik/RQah22IOxF0MRFnIAkkOHoTYjGtcjYlFxMQm3b6WhR3YCZh7ZGKN1CoV5p/AvU3NO9Q=='
bxvals = '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'


def _pretty_space_text(left='', middle='', right='', total_width=77):
    """
    keep position of left and right strings constant regardless of middle width
    default width = 80 based on New Courier and 8 point font

    """
    pad1 = " " * ((total_width - len(middle)) // 2)  # 145 spaces found empirically
    pad2 = " " * ((total_width - len(middle)) % 2)  # deal with odd number of pad1 spaces
    msg1 = left + pad1 + middle + pad1 + pad2 + right
    return msg1


def build_fig(dat, figsize=(11,8.5), dpi=100):

    imin = 2034
    imax = 2355
    minplot = 0.1
    maxplot = 4.9
    xtick_range = [1.,2.,3.,4.]
    nobase = False
    fontname = 'Courier New'
    nsect = [0, 8, 8, 13]
    nfitcol = 4
    fixphase = True
    vespa_version = '1.1.1rc1'
    data_source = 'D:\\bsoher\\code\\repository_svn\\sample_data\\press_cp_svs_data\\press_cp2.rsd'
    viffname = 'Analysis - Fit Tab'
    timestamp = datetime.datetime(*time.localtime()[:6]).isoformat()

    xvals = dat[4]

    #--------------------------------------------------------------------------
    # Create the figure and canvas
    #
    # A canvas must be manually attached to the figure (pyplot would automatically do it).
    # This is done by instantiating the canvas with the figure as argument.

    fig = Figure(figsize=figsize, facecolor='white', dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    fig.subplots_adjust(hspace=0.001)
    nullfmt = NullFormatter()                   # used to suppress labels on an axis
    local_grey = (10./255.,10./255.,10./255.)   # used to tweak font color locally

    # Layout for an LCModel-like landscape report

    left, bottom    = 0.06, 0.07        # set up for 8.5x11 landscape printout
    w1, w2          = 0.52, 0.35        # orig 0.55 0.35
    h1, h2          = 0.73, 0.07
    hpad, vpad      = 0.02, 0.001      # orig 0.001  0.001

    rect1 = [left,         bottom+h1, w1, h2]    # xmin, ymin, dx, and dy
    rect2 = [left,         bottom,    w1, h1]
    rect4 = [left+w1+hpad, bottom, w2, h1+h2+0.015]

    # Noise Residual Plot -----------------------------------------------------

    dat1 = dat[0] # (freq - yfits - base)[imin:imax].real
    min1, max1 = min(dat1),max(dat1)
    delt1 = (max1 - min1)*0.75
    min1, max1 = min1 - delt1, max1 + delt1

    ax1 = fig.add_axes(rect1)
    ax1.xaxis.set_major_formatter(nullfmt)  # no x labels, have to go before plot()
    ax1.plot(xvals, dat1, 'k', lw=1.0)
    ax1.set_xlim(maxplot, minplot)
    ax1.set_ylim(min1, max1)
    ax1.set_xticks(xtick_range)
    ax1.set_yticks([0.0, max1])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    # Data, Fit, Baseline Plot ------------------------------------------------

    dat2  = dat[1] #  freq[imin:imax].real
    dat2b = dat[2] #  (yfits+base)[imin:imax].real
    dat2c = dat[3] #  base[imin:imax].real
    min2, max2 = min(dat2),max(dat2)
    delt2 = (max2 - min2)*0.05
    min2, max2 = min2-delt2, max2+delt2
    tmp = abs(max2) if abs(max2) > abs(min2) else abs(min2)     # in case spectrum flipped
    major2 = tmp/4.0
    if major2 > 2.5:
        major2 = np.round(major2,0)

    ax2 = fig.add_axes(rect2)
    ax2.plot(xvals, dat2,  'k', lw=1.0)
    ax2.plot(xvals, dat2b, 'r', lw=1.5, alpha=0.7)
    if not nobase:
        ax2.plot(xvals, dat2c, 'g', lw=1.0)
    ax2.set_ylabel('Spectral Data, Fit, Baseline', fontsize=8.0)
    ax2.set_xlabel('Chemical Shift [ppm]',         fontsize=8.0)
    ax2.set_ylim(min2, max2)
    ax2.set_xlim(maxplot, minplot)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax2.yaxis.set_major_locator(MultipleLocator(major2))        # this is for even distrib across plot
    ax2.yaxis.set_minor_locator(MultipleLocator(major2*0.5))

    # Common Settings for both plot axes --------------------------------------

    for ax in [ax1,ax2]:
        ax.xaxis.set_major_locator(MultipleLocator(1))        # this is for even distrib across plot
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.grid(which='major', axis='x', linewidth=0.25, linestyle='-', color='0.8')
        ax.grid(which='minor', axis='x', linewidth=0.25, linestyle=(0,(18,6)), color='0.8')
        ax.grid(which='major', axis='y', linewidth=0.25, linestyle=(0,(6,18)), color='0.8')
        # ax.grid(which='minor', axis='y', linewidth=0.25, linestyle=':', color='0.8')


    # Table Setup -------------------------------------------------------------

    ax4 = fig.add_axes(rect4)
    ax4.xaxis.set_major_formatter(nullfmt)      # turn off axis markers
    ax4.yaxis.set_major_formatter(nullfmt)
    ax4.axis('off')

    the_table = ax4.table(cellText=fit_data, cellLoc='left',
                          colLoc='left',   colLabels=None, colWidths=None,
                          rowLoc='center', rowLabels=None,
                          fontsize=7.0, loc='upper center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7.0)
    for item in range(len(fit_data[0])):
        the_table.auto_set_column_width(item)   # mpl bug requires that each col be added individually

    table_props = the_table.properties()
    cheight = table_props['children'][0].get_height()   # all start with same default
    keys = list(table_props['celld'].keys())
    for key in keys:                            # use cell dict to test grid location for line settings
        cell = table_props['celld'][key]
        cell.set_height(1.0*cheight)
        cell.get_text().set_fontname(fontname)
        if key[0] in nsect:                     # this is a header cell
            if key[1] == 0:                     # - leftmost
                cell.visible_edges = 'BLT'
            elif key[1] == nfitcol-1:           # - rightmost
                cell.visible_edges = 'BRT'
            else:
                cell.visible_edges = 'BT'
            cell.set_linewidth(1.0)
            cell.set_linestyle('-')
            cell.get_text().set_fontweight('bold')
        else:                                   # not a header cell
            if key[1] == 0:                     # - leftmost
                cell.visible_edges = 'L'
                if key[0] == len(fit_data)-1:
                    cell.visible_edges = 'BL'
            elif key[1] == nfitcol-1:           # - rightmost
                cell.visible_edges = 'R'
                if key[0] == len(fit_data)-1:
                    cell.visible_edges = 'BR'
            else:
                cell.visible_edges = ''
                if key[0] == len(fit_data)-1:   # last line in table
                    cell.visible_edges = 'B'
            cell.set_linewidth(0.25)
            cell.set_linestyle('-')

    # Retrieve an element of a plot and set properties
    for tick in ax2.xaxis.get_ticklabels():
        tick.set_fontsize(8.0)
        tick.set_fontname(fontname)
        tick.set_color(local_grey)
        tick.set_weight('bold')         # 'normal'

    for ax in [ax1,ax2]:
        ax.xaxis.label.set_fontname(fontname)
        ax.yaxis.label.set_fontname(fontname)
        ax.yaxis.offsetText.set_fontname(fontname)
        ax.yaxis.offsetText.set_fontsize(8.0)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(7.0)
            tick.set_fontname(fontname)
            tick.set_color(local_grey)
            tick.set_weight('bold')
            if ax == ax2:
                tick.set_rotation(90)

    middle = "Fitted - Full Model (Phase0/1 Corrected)" if fixphase else "Fitted - Full Model"
    msg1 = _pretty_space_text("Vespa-Analysis Version: %s" % (vespa_version, ), middle, "Processing Timestamp: %s"   % (timestamp, ))
    fig.text(0.042, 0.94, msg1,
                            wrap=True,
                            horizontalalignment='left',
                            fontsize=8,
                            fontname=fontname)

    msg = "Data Source : %s \nVIFF File   : %s" % (data_source, viffname )
    fig.text(0.042, 0.89,   msg,
                            wrap=True,
                            horizontalalignment='left',
                            color=local_grey,
                            fontsize=8,
                            fontname=fontname)
    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()
    png_buf = np.frombuffer(buf, dtype=np.uint8)

    xdim = int(figsize[0]*dpi)
    ydim = int(figsize[1]*dpi)
    png_buf.shape = xdim, ydim, 1, 1, 1, 4  # x,y,z,cha,images,rgba

    return png_buf