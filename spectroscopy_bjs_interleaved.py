import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter
import matplotlib.pyplot as plt

# Folder for debug output files
debugFolder = "/tmp/share/debug"
 
def process(connection, config, metadata):
    logging.info("bjs-ccx Config: \n%s", config)
 
    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("bjs-ccx Metadata: \n%s", metadata.toxml('utf-8'))
 
        logging.info("bjs-ccx Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("bjs-ccx First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)",
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.y,
            metadata.encoding[0].encodedSpace.matrixSize.z,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)
 
    except:
        logging.info("bjs-ccx Improperly formatted metadata: \n%s", metadata)
 
    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 1
    acqGroup = []
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
                    acqGroup.append(item)
 
                    # process each average, return image that is sent back to the client.
                    logging.info("bjs-ccx Processing a group of k-space data")
                    image = process_raw(acqGroup, connection, config, metadata, series_index=currentSeries)
                    connection.send_image(image)
                    image2 = process_raw(acqGroup, connection, config, metadata, series_index=currentSeries+1)
                    connection.send_image(image2)
                    acqGroup = []
 
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                logging.error("Unsupported data  type %s", type(item).__name__)
 
            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)
 
            elif item is None:
                break
 
            else:
                logging.error("Unsupported data  type %s", type(item).__name__)
 
        # Extract raw ECG waveform data. Basic sorting to make sure that data
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)
 
        # Process any remaining groups of raw or image data.  This can
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("bjs-ccx Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, connection, config, metadata)
            connection.send_image(image)
            acqGroup = []
 
        # if len(imgGroup) > 0:
        #     logging.info("bjs-ccx Processing a group of images (untriggered)")
        #     image = process_image(imgGroup, connection, config, metadata)
        #     connection.send_image(image)
        #     imgGroup = []
 
    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
 
    finally:
        connection.send_close()
 

def process_raw(group, connection, config, metadata, series_index=0):
    # Format data into a [cha RO ave lin seg] array
    nAve = 1 # int(metadata.encoding[0].encodingLimits.average.maximum                - metadata.encoding[0].encodingLimits.average.minimum)                + 1
    nLin = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_1.minimum) + 1
    nSeg = int(metadata.encoding[0].encodingLimits.segment.maximum                - metadata.encoding[0].encodingLimits.segment.minimum)                + 1
    nRO  = mrdhelper.get_userParameterLong_value(metadata, 'SpecVectorSize')

    if nRO is None:
        nRO = int((group[0].data.shape[1] - group[0].discard_pre - group[0].discard_post) / 2)  # 2x readout oversampling
        logging.warning("Could not find SpecVectorSize in header -- using size %d from data", nRO)

    # 2x readout oversampling
    nRO = nRO * 2

    logging.info("bjs-ccx MRD header: %d averages, %d lines, %d segments" % (nAve, nLin, nSeg))

    aves = [0,] # [acquisition.idx.average              for acquisition in group]
    lins = [0,] # [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    segs = [0,] # [acquisition.idx.segment              for acquisition in group]

    if len(set(lins)) > 1 or len(set(segs)) > 1:
        raise ValueError("Method process_raw() given non-SVS data - returning.")

    data = np.zeros((group[0].data.shape[0], 
                     nRO,
                     nAve, 
                     nLin, 
                     nSeg), 
                    group[0].data.dtype)

    for acq, ave, lin, seg in zip(group, aves, lins, segs):
        data[:,:,ave,lin,seg] = acq.data[:,acq.discard_pre:(acq.data.shape[1]-acq.discard_post)]

    logging.info("bjs-ccx Incoming raw spectroscopy data is shape %s" % (data.shape,))

    # Select coil with the best SNR
    indBestCoil = np.argmax(np.mean(np.abs(data[:,:,0,0,0]),axis=(1,)))
    data = data[np.newaxis,indBestCoil,...]

    # Remove readout oversampling
    data = fft.fft(data, axis=1)
    data = np.delete(data, np.arange(int(data.shape[1]*1/4),int(data.shape[1]*3/4)), axis=1)
    data = fft.ifft( data, axis=1)

    # Match Siemens convention of complex conjugate representation
    # data = np.conj(data)

    # Match Siemens data scaling
    data = data * 2**25

    # Combine averages
    data = np.mean(data, axis=2, keepdims=True)

    # Collapse into shape [RO lin seg]
    data = np.squeeze(data)

    # Send data back as complex singles
    data = data.astype(np.complex64)
    data = data.transpose()

    if series_index != 0:
        data = data * 2 * series_index      # slightly modify data by series

    logging.info("bjs-ccx Outgoing spectroscopy data is shape %s" % (data.shape,))

    # Create new MRD instance for the processed image
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
    # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)
    tmpImg = ismrmrd.Image.from_array(data, transpose=False)

    # Set the header information
    tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), group[0].getHead()))

    # User defined series_index
    tmpImg.image_series_index = series_index
    indx_str = str(series_index)

    # Single voxel
    tmpImg.field_of_view = (ctypes.c_float(data.shape[0]*metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    tmpImg.image_index   = 1
    tmpImg.flags         = 2**5   # IMAGE_LAST_IN_AVERAGE

    logging.info("bjs-ccx Outgoing spectroscopy data is field_of_view %s, %s, %s" % (np.double(tmpImg.field_of_view[0]), np.double(tmpImg.field_of_view[1]), np.double(tmpImg.field_of_view[2])))
    logging.info("bjs-ccx Outgoing spectroscopy data is matrix_size   %s, %s, %s" % (tmpImg.getHead().matrix_size[0], tmpImg.getHead().matrix_size[1], tmpImg.getHead().matrix_size[2]))

    # Set ISMRMRD Meta Attributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta['DataRole']                            = 'Spectroscopy'
    tmpMeta['ImageProcessingHistory']              = ['FIRE', 'SPECTRO', 'PYTHON', 'TMP_IMG'+indx_str]
    tmpMeta['Keep_image_geometry']                 = 1
    tmpMeta['SiemensControl_SpectroData']          = ['bool', 'true']
    #tmpMeta['SiemensControl_Suffix4DataFileName']  = ['string', '-1_1_1_1_1_1']

    # Change dwell time to account for removal of readout oversampling
    dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms

    if dwellTime is None:
        logging.error("Could not find DwellTime_0 in MRD header")
    else:
        logging.info("bjs-ccx Found acquisition dwell time from header: " + str(dwellTime*1000))
        tmpMeta['SiemensDicom_RealDwellTime']         = ['int', str(int(dwellTime*1000*2))]

    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    tmpImg.attribute_string = xml

    images = [tmpImg]

    # roiImg = plot_spectra(tmpImg, connection, config, metadata)
    # if roiImg is not None:
    #     images.append(roiImg)

    return images
 





