import ismrmrd
import os
import sys
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

# bjs imports
from logging import FileHandler, Formatter
#from pymidas_inline.epsi.do_epsi import do_init, do_epsi

# BJS_DEBUG_PATH = "D:\\temp\\debug_fire\\"
BJS_DEBUG_PATH = "/tmp/share/debug"
LOG_FORMAT = ('%(asctime)s | %(levelname)s | %(message)s')

# Folder for debug output files
debugFolder = "/tmp/share/debug"
# debugFolder = "D:\\temp\\debug_fire"

logger_bjs = logging.getLogger("bjs_log")
logger_bjs.setLevel(logging.DEBUG)

file_handler = FileHandler(os.path.join(BJS_DEBUG_PATH, 'log_epsi_out.txt'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(Formatter(LOG_FORMAT))
logger_bjs.addHandler(file_handler)


# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.setFormatter(Formatter(LOG_FORMAT))
# logger_bjs.addHandler(stdout_handler)


class BlockEpsi:
    """
    Building block object used to create a list of MIDAS processing blocks.

    This object represents the settings and results involved in processing
    data for EPSI.

    In here we also package all the functionality needed to save and recall
    these values to/from an XML node.

    """
    XML_VERSION = "2.0.0"   # allows us to change XML I/O in the future

    def __init__(self, attributes=None):

        # Settings - algorithm flags and parameters

        self.trajectory_filename    = r"g100_r130_sim.dat"      # expand on run
        self.echo_drift_corr        = 1
        self.frequency_drift_corr   = 2
        self.frequency_drift_value  = 0.000
        self.invert_z               = True
        self.swap_lr                = False
        self.echo_output            = 0
        self.echo_average_fix       = True
        self.retain_input_files     = False
        self.plot_echo_positions    = True
        self.nx_resample            = 50
        self.apply_kx_phase_corr    = 1         # deprecated, always done if echoShiftsOpt set

        # multiprocessing settings
        self.single_process     = True          # use multiprocessing or not
        self.nprocess           = 1             # number of cores to use
        self.chunksize          = None          # alignment with other pymidas modules

        # data information
        self.nt                 = None          # these are defaults for Siemens
        self.nx                 = None
        self.ny                 = 50
        self.nz                 = 18
        self.os                 = 2
        self.os_orig            = 2
        self.sw                 = 2500.0
        self.nx_out             = 50
        self.full_traj          = False          # deprecated?
        self.sampling_interval  = 4000

        # dynamically set
        self.do_setup           = True      # setup arrays first off
        
        self.fovx               = 280.0
        self.fovy               = 280.0
        self.fovz               = 180.0
        self.ncha               = 12
        self.n_process_nodes    = 1
        self.data_id_array      = []
        self.nx_resample        = 50
        self.byte_order         = ''
        self.num_phencs         = 1
        self.nd                 = 1
        self.td                 = 1.0
        self.scan_data          = ''
        self.mrdata_fnames      = []

        self.series_label       = 'SI_REF'
        self.pix_spacing_1      = 5.6
        self.pix_spacing_2      = 5.6
        self.pix_spacing_3      = 10.0
        self.mrdata             = None
        self.out_filename       = ''
        self.out_indx           = 1     # DICOM slice number indexed from 1
        self.save_output        = True
        self.channel            = ''
        self.csa_pad_length     = 0
        self.fin_names          = []
        self.fout_names         = []
        self.echo_shifts_slope  = None
        self.echo_shifts        = None
        self.echo_phases        = None
        self.Is_GE              = False
        self.last_zindx        = 0

        # data storage

        self.metab_init = None
        self.water_init = None
        self.water = []
        self.metab = []

    @property
    def n_channels(self):
        return self.ncha
    @n_channels.setter
    def n_channels(self, value):
        self.ncha = value

    @property
    def nchannels(self):
        return self.ncha
    @nchannels.setter
    def nchannels(self, value):
        self.ncha = value


def process(connection, config, metadata):
    """
    This version is same as the ICE 'Raw' selection. Data is stored frame
    by frame to the DICOM database with just raw k-space data in it.

    Input from server is a group of data from each ADC, this method will
    collate the EPI readouts into one group for Metab signals and another
    group for Water signals that comprise one TR acquisition.  Data will
    be saved to one of two arrays that hold all the (ny, nt, nx) encodes
    for one Z phase encode of data. When that array is full, it will be
    sent back from FIRE for storage in the database. And a new array of
    data will be collated.

        csi_se.cpp code re. encoding indices

        PAR - ZPhase 18 - m_adc1.getMDH().setCpar((short) m_sh_3rd_csi_addr[i] + m_sh_3rd_csi_addr_offset);
        LIN - YPhase 50 - m_adc1.getMDH().setClin(SpecVectorSizeshort) m_sh_2nd_csi_addr[i] + m_sh_2nd_csi_addr_offset);
        ECO - 0/1 Water/Metab m_adc1.getMDH().setCeco(0);     WS vs Water Non-Suppressed
        SEG - 100 (50 w/o OS) m_adc1.getMDH().setCseg(ADCctr + +);        EPI RO segment
        m_adc1.getMDH().setFirstScanInSlice(!i && !j);
        m_adc1.getMDH().setLastScanInSlice(i == (m_lN_csi_encodes - 1) && j == (m_sh_csi_weight[i] - 1));
        m_adc1.getMDH().addToEvalInfoMask (MDH_PHASCOR);

    """

    block = BlockEpsi()

    logging.info("Config: \n%s", config)
 
    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
 
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)",
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.y,
            metadata.encoding[0].encodedSpace.matrixSize.z,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

        block.nz = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_2.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_2.minimum) + 1
        block.ny = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_1.minimum) + 1
        block.nt = mrdhelper.get_userParameterLong_value(metadata, 'SpecVectorSize')
        block.fovx = metadata.encoding[0].reconSpace.fieldOfView_mm.x
        block.fovy = metadata.encoding[0].reconSpace.fieldOfView_mm.y
        block.fovz = metadata.encoding[0].reconSpace.fieldOfView_mm.z
        block.ncha = metadata.acquisitionSystemInformation.receiverChannels
        block.ice_select = mrdhelper.get_userParameterLong_value(metadata, 'EpsiWip_IceSelect')
    except:
        logging.info("Improperly formatted metadata or auxiliary variables: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    acq_group = []
    ref_group = []

    logger_bjs.info("----------------------------------------------------------------------------------------")
    logger_bjs.info("Start EPSI.py run")

    inline_method = 'epsi'  # bjs ['raw', 'epsi'][block.ice_select]   # or 'epsi' or 'both'

    try:
        for item in connection:
            # -------------------------------------------------------------------------------------
            # Raw k-space data messages
            # - user_int[3] is non-zero for all ADCs of final y-encode of a given z-slice
            # - data collate for RAW complete given the following 3 things:
            #   - idx.contrast == 1 (both metab and water data acquired)
            #   - user_int[3] > 0 (this is last y-encode of a given z-slice)
            #   - user_int[1] > 0 (this is last ADC of the EPI acquisition)
            # - data collate for EPSI complete given the following 3 things:
            #   - idx.contrast == 1 (both metab and water data acquired)
            #   - user_int[3] > 0 (this is last y-encode of a given z-slice)
            #   - user_int[1] > 0 (this is last ADC of the EPI acquisition)
            # - data collate for EPSI complete when both user_int[1] AND user_int[3] are non-zero
            # -------------------------------------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                flag_ctr_kspace = item.user_int[2] > 0
                flag_last_epi = item.user_int[1] > 0
                flag_last_yencode = item.user_int[3] > 0    # bjs orig - item.idx.kspace_encode_step_1 == block.ny - 1
                
                zindx = item.idx.kspace_encode_step_2
                yindx = item.idx.kspace_encode_step_1

                if inline_method == 'raw':
                    if block.do_setup:
                        block.ncha, block.nx = item.data.shape
                        dims = [block.nz, block.ny, block.nt, block.nx]
                        block.water = []
                        block.metab = []
                        for i in range(block.ncha):
                            block.water.append(np.zeros(dims, item.data.dtype))
                            block.metab.append(np.zeros(dims, item.data.dtype))
                        block.do_setup = False

                    if flag_ctr_kspace:             # Center of kspace data ignored here
                        pass
                    else:                           # Regular kspace acquisition
                        acq_group.append(item)
                        if flag_last_epi:
                            process_group_raw(block, acq_group, config, metadata)
                            if item.idx.contrast == 1 and flag_last_yencode:
                                logger_bjs.info("**** bjs - send_raw() -- zindx = %d, yindx = %d " % (zindx, yindx))
                                images = send_raw(block, acq_group, connection, config, metadata)
                                connection.send_image(images)
                                block.last_zindx += 1
                            acq_group = []

                elif inline_method == 'epsi':
                    if block.do_setup:
                        block.ncha, block.nx = item.data.shape
                        block.nx2 = int(block.nx // 2)
                        block.nt2 = int(block.nt // 2)

                        dims = [block.nz, block.ny, block.nt, block.nx]
                        block.water = []
                        block.metab = []
                        dims_epsi = [block.nz, block.ny, block.nx2, block.nt2]
                        block.water_epsi = []
                        block.metab_epsi = []
                        for i in range(block.ncha):
                            block.water.append(np.zeros(dims, item.data.dtype))
                            block.metab.append(np.zeros(dims, item.data.dtype))
                            block.water_epsi.append(np.zeros(dims_epsi, item.data.dtype))
                            block.metab_epsi.append(np.zeros(dims_epsi, item.data.dtype))
                        block.do_setup = False
                        block.ref_done = False

                    if flag_ctr_kspace:             # Center of kspace data
                        ref_group.append(item)
                        if item.idx.contrast == 1 and flag_last_epi:
                            process_init_epsi(block, ref_group, config, metadata)
                            ref_group = []
                            block_ref_done = True
                    else:                           # Regular kspace acquisition
                        acq_group.append(item)
                        if flag_last_epi:
                            process_raw_to_epsi(block, acq_group, config, metadata)
                            if item.idx.contrast == 1 and flag_last_yencode:
                                logger_bjs.info("**** bjs - send_raw() -- zindx = %d, yindx = %d " % (zindx, yindx))
                                images = send_epsi(block, acq_group, connection, config, metadata)
                                connection.send_image(images)
                                block.last_zindx += 1
                            acq_group = []
                else:
                    msg = "Inlne process method not recognized: %s", inline_method
                    logging.error(msg)
                    raise ValueError(msg)

            elif item is None:
                break
 
            else:
                logging.error("Unsupported data  type %s", type(item).__name__)
 
    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
 
    finally:
        connection.send_close()


def process_group_raw(block, group, config, metadata):

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = list(range(block.nt))

    if len(set(indz)) > 1:
        logger_bjs.info("Too many Z encodes in TR data group")
    if len(set(indy)) > 1:
        logger_bjs.info("Too many Y encodes in TR data group")

    for acq, iz, iy, it in zip(group, indz, indy, indt):
        for i in range(block.ncha):
            if group[0].idx.contrast == 0:
                block.metab[i][iz, iy, it, :] = acq.data[i,:]
            else:
                block.water[i][iz, iy, it, :] = acq.data[i,:]
    return


def process_init_epsi(block, group, config, metadata):
    """ Format data into a [cha RO ave lin seg] array """

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = list(range(block.nt))

    if len(set(indz)) > 1:
        logger_bjs.info("Too many Z encodes in Init data group")
    if len(set(indy)) > 1:
        logger_bjs.info("Too many Y encodes in Init data group")

    for acq, iz, iy, it in zip(group, indz, indy, indt):
        for i in range(block.ncha):
            if group[0].idx.contrast == 0:
                block.metab[i][iz, iy, it, :] = acq.data[i,:]
            else:
                block.water[i][iz, iy, it, :] = acq.data[i,:]

    # Ref acq should be in block.water/metab[:][0,0,:,:] arrays

    nt, nx, ny, nz, nchan = block.nt, block.nx, block.ny, block.nz, block.n_channels
    dims = (nz, ny, nt, nx)

    k_data = inline_init_traj_corr(block)       # TODO bjs - do I need to save k_data

    # Preparation section

    for ichan in range(nchan):
        xino, xine = init_interp_kx(block, k_data, ichan)
        expo, expe = init_process_kt(block, ichan, reverse=False)
        block.xino[ichan][:,:] = xino
        block.xine[ichan][:,:] = xine
        block.expo[ichan][:,:] = expo
        block.expe[ichan][:,:] = expe

    block.ref_done = True

        

def process_raw_to_epsi(block, group, config, metadata):

    indz = set([item.idx.kspace_encode_step_2 for item in group])
    indy = set([item.idx.kspace_encode_step_1 for item in group])
    indt = set([item.idx.segment for item in group])
    #indt = list(range(block.nt))
    ieco = group[0].idx.contrast

    if len(indz) > 1:
        logger_bjs.info("Too many Z encodes in TR data group")
    if len(indy) > 1:
        logger_bjs.info("Too many Y encodes in TR data group")

    for acq, it in zip(group, indt):
        for i in range(block.ncha):
            if group[0].idx.contrast == 0:
                block.metab[i][indz, indy, it, :] = acq.data[i,:]
            else:
                block.water[i][indz, indy, it, :] = acq.data[i,:]

    # data_out = do_epsi_process(block, indy, indz, ieco)
    #
    # for i in range(block.ncha):
    #     if ieco == 0:
    #         block.metab_epsi[i][indz, indy, :, :] = data_out[i, :, :]  # dims should be cha,x,t here
    #     else:
    #         block.water_epsi[i][indz, indy, :, :] = data_out[i, :, :]  # dims should be cha,x,t here

    return


def send_raw(block, group, connection, config, metadata):

    zindx = block.last_zindx   # TODO bjs - what range do we take?
    images = []

    metadata.encoding[0].echoTrainLength = 2

    slthick = float(block.fovz/block.nz)
    posoff = slthick * (zindx - 0.5 * block.nz + .5)
    norm_sag = mrdhelper.get_userParameterDouble_value(metadata, 'EpsiWip_sNormal_dSag')
    norm_cor = mrdhelper.get_userParameterDouble_value(metadata, 'EpsiWip_sNormal_dCor')
    norm_tra = mrdhelper.get_userParameterDouble_value(metadata, 'EpsiWip_sNormal_dTra')
    pos_sag = mrdhelper.get_userParameterDouble_value(metadata, 'EpsiWip_sPosition_dSag')
    pos_cor = mrdhelper.get_userParameterDouble_value(metadata, 'EpsiWip_sPosition_dCor')
    pos_tra = mrdhelper.get_userParameterDouble_value(metadata, 'EpsiWip_sPosition_dTra')

    posvecx = pos_sag + norm_sag * posoff
    posvecy = pos_cor + norm_cor * posoff
    posvecz = pos_tra + norm_tra * posoff


    # Set ISMRMRD Meta Attributes
    tmpMetaMet = ismrmrd.Meta()
    tmpMetaMet['DataRole'] = 'Spectroscopy'
    tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON']
    tmpMetaMet['Keep_image_geometry'] = 1
    tmpMetaMet['SiemensControl_SpectroData'] = ['bool', 'true']
    tmpMetaMet['InternalSend'] = 1      # skips SpecSend functor in ICE program - might keep header from modification?

    tmpMetaMet['SiemensDicom_EchoTrainLength'] = 2
    tmpMetaMet['SiemensDicom_SequenceDescription'] = str(metadata.measurementInformation.protocolName)+'_FIRE_RAW'
    tmpMetaMet['SiemensDicom_PercentPhaseFoV'] = 1.0
    tmpMetaMet['SiemensDicom_PercentSampling'] = 1.0
    tmpMetaMet['SiemensDicom_NoOfCols'] = int(block.nx)
    tmpMetaMet['SiemensDicom_SliceThickness'] = slthick
    tmpMetaMet['SiemensDicom_ProtocolSliceNumber'] = zindx
    tmpMetaMet['SiemensDicom_TE'] = metadata.sequenceParameters.TE[0]
    tmpMetaMet['SiemensDicom_TR'] = metadata.sequenceParameters.TR[0]
    tmpMetaMet['SiemensDicom_TI'] = metadata.sequenceParameters.TI[0]
    tmpMetaMet['SiemensDicom_PixelSpacing'] = [float(block.fovx / block.nx), float(block.fovy / block.ny)]

    tmpMetaWat = ismrmrd.Meta()
    tmpMetaWat['DataRole'] = 'Spectroscopy'
    tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON']
    tmpMetaWat['Keep_image_geometry'] = 1
    tmpMetaWat['SiemensControl_SpectroData'] = ['bool', 'true']
    tmpMetaWat['InternalSend'] = 1

    tmpMetaWat['SiemensDicom_EchoTrainLength'] = 2
    tmpMetaWat['SiemensDicom_SequenceDescription'] = str(metadata.measurementInformation.protocolName)+'FIRE_RAW'
    tmpMetaWat['SiemensDicom_PercentPhaseFoV'] = 1.0
    tmpMetaWat['SiemensDicom_PercentSampling'] = 1.0
    tmpMetaWat['SiemensDicom_NoOfCols'] = int(block.nx)
    tmpMetaWat['SiemensDicom_SliceThickness'] = slthick
    tmpMetaWat['SiemensDicom_ProtocolSliceNumber'] = zindx
    tmpMetaWat['SiemensDicom_TE'] = metadata.sequenceParameters.TE[1]
    tmpMetaWat['SiemensDicom_TR'] = metadata.sequenceParameters.TR[1]
    tmpMetaWat['SiemensDicom_TI'] = metadata.sequenceParameters.TI[0]
    tmpMetaWat['SiemensDicom_PixelSpacing'] = [float(block.fovx / block.nx), float(block.fovy / block.ny)]


    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)



    # # NOT NEEDED FOR RAW SAVE, JUST FOR PROCESSED EPSI
    # # Change dwell time to account for removal of readout oversampling
    # dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms
    #
    # if dwellTime is None:
    #     logging.error("Could not find DwellTime_0 in MRD header")
    # else:
    #     logging.info("Found acquisition dwell time from header: " + str(dwellTime * 1000))
    #     tmpMetaMet['SiemensDicom_RealDwellTime'] = ['int', str(int(dwellTime * 1000 * 2))]

    # FROM BARTFIRE
    # # Add image orientation directions to MetaAttributes if not already present
    # if tmpMetaMet.get('ImageRowDir') is None:
    #     tmpMetaMet['ImageRowDir'] = ["{:.18f}".format(tmpImg.getHead().read_dir[0]),
    #                               "{:.18f}".format(tmpImg.getHead().read_dir[1]),
    #                               "{:.18f}".format(tmpImg.getHead().read_dir[2])]
    #
    # if tmpMetaMet.get('ImageColumnDir') is None:
    #     tmpMetaMet['ImageColumnDir'] = ["{:.18f}".format(tmpImg.getHead().phase_dir[0]),
    #                                  "{:.18f}".format(tmpImg.getHead().phase_dir[1]),
    #                                  "{:.18f}".format(tmpImg.getHead().phase_dir[2])]


    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab[icha][zindx, :,:,:].copy()      # TODO bjs - what range do we take?
        water = block.water[icha][zindx, :,:,:].copy()

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1] * ms[2]
        water.shape = ws[0], ws[1] * ws[2]

        # metab = np.conj(metab)
        # water = np.conj(water)

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.image_index = block.out_indx + 0
        tmpImgWat.image_index = block.out_indx + 1
        block.out_indx += 2

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

    return images


def send_epsi(block, group, connection, config, metadata):

    # NOT DONE -- NOT DONE -- NOT DONE -- NOT DONE -- NOT DONE -- NOT DONE --

    # May have to wait until all data processed to do these 2 steps, then
    # send all data to database??? bjs  From do_epsi.do_epsi()

    nt, nx, ny, nz = block.nt, block.nx, block.ny, block.nz
    nx2 = int(nx / 2)  # size of regridded data
    nt2 = int(nt / 2)

    if block.set.swap_lr:
        for ichan in range(block.ncha):
            for z in range(nz):
                for x in range(nx2):
                    for t in range(nt2):
                        tmp = np.fliplr(np.squeeze(block.water[ichan][z, :, x, t]))
                        tmp[0] = 0 + 0j     # bjs - may not need this?
                        block.water[ichan][z, :, x, t] = tmp

                        tmp = np.fliplr(np.squeeze(block.metab[ichan][z, :, x, t]))
                        tmp[0] = 0 + 0j     # bjs - may not need this?
                        block.metab[ichan][z, :, x, t] = tmp


    if block.set.invert_z:
        for ichan in range(block.ncha):
            block.water[ichan][:, :, :, :] = block.water[ichan][::-1, :, :, :]
            block.metab[ichan][:, :, :, :] = block.metab[ichan][::-1, :, :, :]

    zindx = block.last_zindx        # TODO bjs - what range do we take?
    images = []

    # Set ISMRMRD Meta Attributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta['DataRole'] = 'Spectroscopy'
    tmpMeta['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON']
    tmpMeta['Keep_image_geometry'] = 1
    tmpMeta['SiemensControl_SpectroData'] = ['bool', 'true']

    tmpMeta['SiemensDicom_SequenceDescription'] = str(metadata.measurementInformation.protocolName)+'FIRE_PROC1'    # for epsi processing ?

    # Change dwell time to account for removal of readout oversampling
    dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms

    # if dwellTime is None:
    #     logging.error("Could not find DwellTime_0 in MRD header")
    # else:
    #     logging.info("Found acquisition dwell time from header: " + str(dwellTime * 1000))
    #     tmpMeta['SiemensDicom_RealDwellTime'] = ['int', str(int(dwellTime * 1000 * 2))]

    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab[icha][zindx, :,:,:].copy()      # TODO bjs - what range do we take?
        water = block.water[icha][zindx, :,:,:].copy()

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1] * ms[2]
        water.shape = ws[0], ws[1] * ws[2]

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.image_index = 1
        tmpImgWat.image_index = 1

        tmpImgMet.attribute_string = xml
        tmpImgWat.attribute_string = xml

        images.append(tmpImgMet)
        images.append(tmpImgWat)

    return images


def dump_flags(item):
    lines = []
    lines.append("ACQ_IS_DUMMYSCAN_DATA                  = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)))
    lines.append("ACQ_LAST_IN_ENCODE_STEP1               = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)))
    lines.append("ACQ_FIRST_IN_ENCODE_STEP2              = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2)))
    lines.append("ACQ_LAST_IN_ENCODE_STEP2               = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2)))
    lines.append("ACQ_FIRST_IN_AVERAGE                   = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_AVERAGE)))
    lines.append("ACQ_LAST_IN_AVERAGE                    = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_AVERAGE)))
    lines.append("ACQ_FIRST_IN_SLICE                     = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SLICE)))
    lines.append("ACQ_LAST_IN_SLICE                      = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)))
    lines.append("ACQ_FIRST_IN_CONTRAST                  = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_CONTRAST)))
    lines.append("ACQ_LAST_IN_CONTRAST                   = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_CONTRAST)))
    lines.append("ACQ_FIRST_IN_PHASE                     = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_PHASE)))
    lines.append("ACQ_LAST_IN_PHASE                      = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_PHASE)))
    lines.append("ACQ_FIRST_IN_REPETITION                = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_REPETITION)))
    lines.append("ACQ_LAST_IN_REPETITION                 = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION)))
    lines.append("ACQ_FIRST_IN_SET                       = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SET)))
    lines.append("ACQ_LAST_IN_SET                        = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_SET)))
    lines.append("ACQ_FIRST_IN_SEGMENT                   = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SEGMENT)))
    lines.append("ACQ_LAST_IN_SEGMENT                    = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_SEGMENT)))
    lines.append("ACQ_IS_NOISE_MEASUREMENT               = " + str(item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)))
    lines.append("ACQ_IS_PARALLEL_CALIBRATION            = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)))
    lines.append("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING= " + str(item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)))
    lines.append("ACQ_IS_REVERSE                         = " + str(item.is_flag_set(ismrmrd.ACQ_IS_REVERSE)))
    lines.append("ACQ_IS_NAVIGATION_DATA                 = " + str(item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)))
    lines.append("ACQ_IS_PHASECORR_DATA                  = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)))
    lines.append("ACQ_LAST_IN_MEASUREMENT                = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT)))
    lines.append("ACQ_IS_HPFEEDBACK_DATA                 = " + str(item.is_flag_set(ismrmrd.ACQ_IS_HPFEEDBACK_DATA)))
    lines.append("ACQ_IS_DUMMYSCAN_DATA                  = " + str(item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)))
    lines.append("ACQ_IS_RTFEEDBACK_DATA                 = " + str(item.is_flag_set(ismrmrd.ACQ_IS_RTFEEDBACK_DATA)))
    lines.append("ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA  = " + str(item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA)))
    lines.append("ACQ_IS_PHASE_STABILIZATION_REFERENCE   = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE)))
    lines.append("ACQ_IS_PHASE_STABILIZATION             = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION)))
    lines.append("ACQ_COMPRESSION1                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION1)))
    lines.append("ACQ_COMPRESSION2                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION2)))
    lines.append("ACQ_COMPRESSION3                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION3)))
    lines.append("ACQ_COMPRESSION4                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION4)))
    lines.append("ACQ_USER1                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER1)))
    lines.append("ACQ_USER2                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER2)))
    lines.append("ACQ_USER3                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER3)))
    lines.append("ACQ_USER4                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER4)))
    lines.append("ACQ_USER5                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER5)))
    lines.append("ACQ_USER6                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER6)))
    lines.append("ACQ_USER7                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER7)))
    lines.append("ACQ_USER8                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER8)))

    lines = "\n".join(lines)
    print(lines)

def dump_active_flags(item, prnt=False):
    lines = []

    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1): lines.append("ACQ_IS_DUMMYSCAN_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1):	lines.append("ACQ_LAST_IN_ENCODE_STEP1               = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2):	lines.append("ACQ_FIRST_IN_ENCODE_STEP2              = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2):	lines.append("ACQ_LAST_IN_ENCODE_STEP2               = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_AVERAGE):		lines.append("ACQ_FIRST_IN_AVERAGE                   = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_AVERAGE):		lines.append("ACQ_LAST_IN_AVERAGE                    = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SLICE):		lines.append("ACQ_FIRST_IN_SLICE                     = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):		    lines.append("ACQ_LAST_IN_SLICE                      = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_CONTRAST):		lines.append("ACQ_FIRST_IN_CONTRAST                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_CONTRAST):		lines.append("ACQ_LAST_IN_CONTRAST                   = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_PHASE):		lines.append("ACQ_FIRST_IN_PHASE                     = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_PHASE):		    lines.append("ACQ_LAST_IN_PHASE                      = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_REPETITION):	lines.append("ACQ_FIRST_IN_REPETITION                = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):	lines.append("ACQ_LAST_IN_REPETITION                 = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SET):		    lines.append("ACQ_FIRST_IN_SET                       = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SET):			lines.append("ACQ_LAST_IN_SET                        = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SEGMENT):		lines.append("ACQ_FIRST_IN_SEGMENT                   = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SEGMENT):		lines.append("ACQ_LAST_IN_SEGMENT                    = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):	lines.append("ACQ_IS_NOISE_MEASUREMENT               = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION): lines.append("ACQ_IS_PARALLEL_CALIBRATION          = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING): lines.append("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING= True")
    if item.is_flag_set(ismrmrd.ACQ_IS_REVERSE):			lines.append("ACQ_IS_REVERSE                         = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):	lines.append("ACQ_IS_NAVIGATION_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):		lines.append("ACQ_IS_PHASECORR_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):	lines.append("ACQ_LAST_IN_MEASUREMENT                = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_HPFEEDBACK_DATA):	lines.append("ACQ_IS_HPFEEDBACK_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):		lines.append("ACQ_IS_DUMMYSCAN_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_RTFEEDBACK_DATA):	lines.append("ACQ_IS_RTFEEDBACK_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA): lines.append("ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE):lines.append("ACQ_IS_PHASE_STABILIZATION_REFERENCE   = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION):lines.append("ACQ_IS_PHASE_STABILIZATION             = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION1):		    lines.append("ACQ_COMPRESSION1                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION2):		    lines.append("ACQ_COMPRESSION2                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION3):		    lines.append("ACQ_COMPRESSION3                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION4):		    lines.append("ACQ_COMPRESSION4                       = True")
    if item.is_flag_set(ismrmrd.ACQ_USER1):			        lines.append("ACQ_USER1                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER2):			        lines.append("ACQ_USER2                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER3):		    	    lines.append("ACQ_USER3                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER4):		    	    lines.append("ACQ_USER4                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER5):		    	    lines.append("ACQ_USER5                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER6):		    	    lines.append("ACQ_USER6                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER7):			        lines.append("ACQ_USER7                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER8):			        lines.append("ACQ_USER8                              = True")

    if lines == []:
        lines = 'No active flags.'
    else:
        lines = "\n".join(lines)

    if prnt == True:
        print(lines)
    else:
        return lines

