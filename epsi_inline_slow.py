
# Copyright (c) 2024-2025 Brian J Soher - All Rights Reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without explicit permission.

import ismrmrd
import os
import logging
import traceback
import numpy as np
import ctypes
import mrdhelper
import constants

from epsi_inline_util import inline_init_traj_corr, inline_init_interp_kx, inline_init_process_kt, inline_do_epsi

# bjs imports
from logging import FileHandler, Formatter

#BJS_DEBUG_PATH = "debug_fire"
BJS_DEBUG_PATH = "/tmp/share/debug"
LOG_FORMAT = ('%(asctime)s | %(levelname)s | %(message)s')

# Folder for debug output files
debugFolder = "/tmp/share/debug"
#debugFolder = "D:\\tmp\\debug_fire"

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
        self.sampling_interval  = 2e-6

        # dynamically set
        self.do_setup_raw       = True      # setup arrays first off
        self.do_setup_epsi      = True      # setup arrays first off

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
        self.out_indx_raw       = 1     # DICOM slice number indexed from 1
        self.out_indx_epsi      = 1     # DICOM slice number indexed from 1
        self.save_output        = True
        self.channel            = ''
        self.csa_pad_length     = 0
        self.fin_names          = []
        self.fout_names         = []
        self.echo_shifts_slope  = None
        self.echo_shifts        = None
        self.echo_phases        = None
        self.Is_GE              = False
        self.last_zindx_full         = 0
        self.last_zindx_part         = 0
        self.last_zindx_epsi_full    = 0
        self.last_zindx_epsi_part    = 0

        self.curr_yindx = 0
        self.curr_zindx = 0

        # data storage

        self.ref = None
        self.water = None
        self.metab = None
        self.water_epsi = None
        self.metab_epsi = None

        self.ref_full = None
        self.ref_part = None
        self.water_raw_full = []
        self.water_raw_part = []
        self.metab_raw_full = []
        self.metab_raw_part = None
        self.water_full = None      # raw data for EPSI processing - need duplicate here in case 'both' processing done
        self.water_part = None      # ditto next three
        self.metab_full = None
        self.metab_part = None
        self.blank_slice = None
        self.water_epsi_full = None
        self.water_epsi_part = None
        self.metab_epsi_full = None
        self.metab_epsi_part = None
        self.blank_epsi_slice = None

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
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3",
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
    acq_group_raw = []
    acq_group_epsi = []
    ref_group = []

    acq_group_raw_full = []
    acq_group_raw_part = []
    acq_group_epsi_full = []
    acq_group_epsi_part = []
    ref_group_full = []
    ref_group_part = []

    logger_bjs.info("----------------------------------------------------------------------------------------")
    logger_bjs.info("Start EPSI_SLOW.py run")

    # if block.ice_select in [0,4]:
    #     inline_method = 'epsi'
    # elif block.ice_select in [0,3]:
    #     inline_method = 'raw'
    # elif block.ice_select == 2:
    #     inline_method = 'both'
    # else:
    #     block.ice_select = 'raw'

    inline_method = 'both'

    ser_num_raw_full = 0
    ser_num_raw_part = 1
    ser_num_epsi_full = 2
    ser_num_epsi_part = 3

    dgb_raw_full = []
    dgb_raw_part = []
    dgb_epsi_full = []
    dgb_epsi_part = []

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

                if inline_method not in ['raw', 'epsi', 'both', 'test']:
                    msg = "Inlne process method not recognized: %s", inline_method
                    logging.error(msg)
                    raise ValueError(msg)

                flag_ctr_kspace = item.user_int[2] > 0
                flag_last_epi = item.user_int[1] > 0
                flag_last_yencode = item.user_int[3] > 0    # bjs orig - item.idx.kspace_encode_step_1 == block.ny - 1

                zindx = item.idx.kspace_encode_step_2
                yindx = item.idx.kspace_encode_step_1
                iset = item.idx.set

                if inline_method in ['raw','both']:
                    if block.do_setup_raw:
                        block.ncha, block.nx = item.data.shape
                        dims = [block.nz, block.ny, block.nt, block.nx]
                        block.water_raw_full = []
                        block.water_raw_part = []   # bjs, using list of arrays to minimize big memory chunks
                        block.metab_raw_full = []
                        block.metab_raw_part = []
                        for i in range(block.ncha):
                            block.water_raw_full.append(np.zeros(dims, item.data.dtype))
                            block.water_raw_part.append(np.zeros(dims, item.data.dtype))
                            block.metab_raw_full.append(np.zeros(dims, item.data.dtype))
                            block.metab_raw_part.append(np.zeros(dims, item.data.dtype))
                        block.blank_slice = np.zeros(dims, item.data.dtype)

                        block.do_setup_raw = False
                        block.sampling_interval = item.sample_time_us * 1e-6

                    if flag_ctr_kspace:             # Center of kspace data ignored here
                        pass
                    else:                           # Regular kspace acquisition
                        if iset == 0:
                            acq_group_raw_full.append(item)
                        else:
                            acq_group_raw_part.append(item)

                        if flag_last_epi:
                            if iset == 0:
                                process_group_raw_full(block, acq_group_raw_full, config, metadata)
                                if item.idx.contrast == 1 and flag_last_yencode:

                                    if block.last_zindx_full < item.idx.kspace_encode_step_2:
                                        while block.last_zindx_full < item.idx.kspace_encode_step_2:
                                            # assume there was a blank slice that we've missed
                                            images = send_raw_blank(block, acq_group_raw_full, metadata, ser_num_raw_full, block.last_zindx_full, flag_full=True)
                                            connection.send_image(images)
                                            block.last_zindx_full += 1

                                    # BJS - check if the last_zindx_xxx is same as zindex in last item?
                                    #       if not, then have to send a blank slice to out and bump the
                                    #       last_zindx_xx until it is same, then continue.

                                    logger_bjs.info("**** bjs - send_raw_full() -- zindx = %d, yindx = %d iset = %d" % (zindx, yindx, iset))
                                    images = send_raw_full(block, acq_group_raw_full, metadata, ser_num_raw_full)
                                    connection.send_image(images)
                                    block.last_zindx_full += 1
                                    for i in range(block.ncha):
                                        block.water_raw_full[i] = block.water_raw_full[i] * 0.0
                                        block.metab_raw_full[i] = block.metab_raw_full[i] * 0.0
                                acq_group_raw_full = []
                            else:
                                process_group_raw_part(block, acq_group_raw_part, config, metadata)
                                if item.idx.contrast == 1 and flag_last_yencode:

                                    if block.last_zindx_part < item.idx.kspace_encode_step_2:
                                        while block.last_zindx_part < item.idx.kspace_encode_step_2:
                                            # assume there was a blank slice that we've missed
                                            images = send_raw_blank(block, acq_group_raw_part, metadata, ser_num_raw_part, block.last_zindx_part, flag_full=False)
                                            connection.send_image(images)
                                            block.last_zindx_part += 1

                                    # BJS - check if the last_zindx_xxx is same as zindex in last item?
                                    #       if not, then have to send a blank slice to out and bump the
                                    #       last_zindx_xx until it is same, then continue.

                                    logger_bjs.info("**** bjs - send_raw_part() -- zindx = %d, yindx = %d iset = %d" % (zindx, yindx, iset))
                                    images = send_raw_part(block, acq_group_raw_part, metadata, ser_num_raw_part)
                                    connection.send_image(images)
                                    block.last_zindx_part += 1
                                    for i in range(block.ncha):
                                        block.water_raw_part[i] = block.water_raw_part[i] * 0.0
                                        block.metab_raw_part[i] = block.metab_raw_part[i] * 0.0
                                acq_group_raw_part = []


                if inline_method in ['epsi','both']:
                    if block.do_setup_epsi:
                        block.ncha, block.nx = item.data.shape
                        block.nx2 = int(block.nx // 2)
                        block.nt2 = int(block.nt // 2)
                        dims = [block.ncha, block.nt, block.nx]
                        dims_epsi = [block.ny, block.ncha, block.nx2, block.nt2]
                        block.ref = np.zeros(dims, item.data.dtype)
                        block.ref_part = np.zeros(dims, item.data.dtype)        # temp storage as only FULL used to init
                        block.water_full = np.zeros(dims, item.data.dtype)
                        block.water_part = np.zeros(dims, item.data.dtype)
                        block.metab_full = np.zeros(dims, item.data.dtype)
                        block.metab_part = np.zeros(dims, item.data.dtype)
                        block.water_epsi_full = np.zeros(dims_epsi, item.data.dtype)
                        block.water_epsi_part = np.zeros(dims_epsi, item.data.dtype)
                        block.metab_epsi_full = np.zeros(dims_epsi, item.data.dtype)
                        block.metab_epsi_part = np.zeros(dims_epsi, item.data.dtype)
                        block.blank_epsi_slice = np.zeros(dims_epsi, item.data.dtype)
                        block.do_setup_epsi = False
                        block.ref_done = False
                        block.sampling_interval = item.sample_time_us * 1e-6

                    block.curr_yindx = yindx
                    block.curr_zindx = zindx

                    if flag_ctr_kspace:             # Center of kspace data
                        if iset == 0:
                            ref_group_full.append(item)
                            if item.idx.contrast == 1 and flag_last_epi:
                                process_init_epsi_full(block, ref_group_full, config, metadata)
                                ref_group_full = []
                        else:
                            ref_group_part.append(item)
                            if item.idx.contrast == 1 and flag_last_epi:
                                process_init_epsi_part(block, ref_group_part, config, metadata)
                                ref_group_part = []

                    else:                           # Regular kspace acquisition
                        if iset == 0:
                            acq_group_epsi_full.append(item)
                        else:
                            acq_group_epsi_part.append(item)

                        if flag_last_epi:
                            if iset == 0:
                                process_raw_to_epsi_full(block, acq_group_epsi_full)
                                if item.idx.contrast == 1 and flag_last_yencode:

                                    if block.last_zindx_epsi_full < item.idx.kspace_encode_step_2:
                                        while block.last_zindx_epsi_full < item.idx.kspace_encode_step_2:
                                            # assume there was a blank slice that we've missed
                                            images = send_epsi_blank(block, acq_group_epsi_full, metadata, ser_num_epsi_full, block.last_zindx_epsi_full, flag_full=True)
                                            connection.send_image(images)
                                            block.last_zindx_epsi_full += 1

                                    # BJS - check if the last_zindx_xxx is same as zindex in last item?
                                    #       if not, then have to send a blank slice to out and bump the
                                    #       last_zindx_xx until it is same, then continue.

                                    logger_bjs.info("**** bjs - send_epsi_full() -- zindx = %d, yindx = %d iset = %d" % (zindx, yindx, iset))

                                    if block.swap_lr:
                                        for ichan in range(block.ncha):
                                            for x in range(int(block.nx / 2)):
                                                for t in range(int(block.nt / 2)):
                                                    tmp = np.fliplr(np.squeeze(block.water_epsi_full[:, ichan, x, t]))
                                                    tmp[0] = 0 + 0j  # bjs - may not need this?
                                                    block.water_epsi_full[:, ichan, x, t] = tmp

                                                    tmp = np.fliplr(np.squeeze(block.metab_epsi_full[:, ichan, x, t]))
                                                    tmp[0] = 0 + 0j  # bjs - may not need this?
                                                    block.metab_epsi_full[:, ichan, x, t] = tmp


                                    images = send_epsi_full(block, acq_group_epsi_full, metadata, ser_num_epsi_full)
                                    connection.send_image(images)
                                    block.last_zindx_epsi_full += 1
                                    block.water_full = block.water_full * 0.0
                                    block.metab_full = block.metab_full * 0.0
                                    block.water_epsi_full = block.water_epsi_full * 0.0
                                    block.metab_epsi_full = block.metab_epsi_full * 0.0
                                acq_group_epsi_full = []

                            else:
                                process_raw_to_epsi_part(block, acq_group_epsi_part)
                                if item.idx.contrast == 1 and flag_last_yencode:

                                    if block.last_zindx_epsi_part < item.idx.kspace_encode_step_2:
                                        while block.last_zindx_epsi_part < item.idx.kspace_encode_step_2:
                                            # assume there was a blank slice that we've missed
                                            images = send_epsi_blank(block, acq_group_epsi_part, metadata, ser_num_epsi_part, block.last_zindx_epsi_part, flag_full=False)
                                            connection.send_image(images)
                                            block.last_zindx_epsi_part += 1

                                    # BJS - check if the last_zindx_xxx is same as zindex in last item?
                                    #       if not, then have to send a blank slice to out and bump the
                                    #       last_zindx_xx until it is same, then continue.

                                    logger_bjs.info("**** bjs - send_epsi_part() -- zindx = %d, yindx = %d iset = %d" % (zindx, yindx, iset))

                                    if block.swap_lr:
                                        for ichan in range(block.ncha):
                                            for x in range(int(block.nx / 2)):
                                                for t in range(int(block.nt / 2)):
                                                    tmp = np.fliplr(np.squeeze(block.water_epsi_part[:, ichan, x, t]))
                                                    tmp[0] = 0 + 0j  # bjs - may not need this?
                                                    block.water_epsi_part[:, ichan, x, t] = tmp

                                                    tmp = np.fliplr(np.squeeze(block.metab_epsi_part[:, ichan, x, t]))
                                                    tmp[0] = 0 + 0j  # bjs - may not need this?
                                                    block.metab_epsi_part[:, ichan, x, t] = tmp

                                    images = send_epsi_part(block, acq_group_epsi_part, metadata, ser_num_epsi_part)
                                    connection.send_image(images)
                                    block.last_zindx_epsi_part += 1
                                    block.water_part = block.water_part * 0.0
                                    block.metab_part = block.metab_part * 0.0
                                    block.water_epsi_part = block.water_epsi_part * 0.0
                                    block.metab_epsi_part = block.metab_epsi_part * 0.0
                                acq_group_epsi_part = []

            elif item is None:
                break

            else:
                logging.error("Unsupported data  type %s", type(item).__name__)

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()


def process_group_raw_full(block, group, config, metadata):

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = list(range(block.nt))

    if len(set(indz)) > 1:
        logger_bjs.info("process_group_raw_full(): Too many Z encodes in TR data group")
    if len(set(indy)) > 1:
        logger_bjs.info("process_group_raw_full(): Too many Y encodes in TR data group")

    for item, iz, iy, it in zip(group, indz, indy, indt):
        for i in range(block.ncha):
            if group[0].idx.contrast == 0:
                block.metab_raw_full[i][iz, iy, it, :] = item.data[i,:]
            else:
                block.water_raw_full[i][iz, iy, it, :] = item.data[i,:]
    return


def process_group_raw_part(block, group, config, metadata):

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = list(range(block.nt))

    if len(set(indz)) > 1:
        logger_bjs.info("process_group_raw_part(): Too many Z encodes in TR data group")
    if len(set(indy)) > 1:
        logger_bjs.info("process_group_raw_part(): Too many Y encodes in TR data group")

    for item, iz, iy, it in zip(group, indz, indy, indt):
        for i in range(block.ncha):
            if group[0].idx.contrast == 0:
                block.metab_raw_part[i][iz, iy, it, :] = item.data[i,:]
            else:
                block.water_raw_part[i][iz, iy, it, :] = item.data[i,:]
    return


def process_init_epsi_full(block, group, config, metadata):
    """ Format data into a [cha RO ave lin seg] array """

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = [item.idx.segment for item in group]

    if len(set(indz)) > 1:
        logger_bjs.info("Too many Z encodes in Init data group")
    if len(set(indy)) > 1:
        logger_bjs.info("Too many Y encodes in Init data group")
    if len(set(indt)) != block.nt * 2:
        logger_bjs.info("Length of segment encodes list not equal to Nt in Init data group")

    for item in group:
        if item.idx.contrast == 1:
            for icha in range(block.ncha):
                it = item.idx.segment % block.nt
                block.ref[icha, it, :] = item.data[icha,:]
                # NB. bjs, only using FULL ref data to init xino, xine, expo, expe

    # Ref acq should be in block.water/metab[:][0,0,:,:] arrays

    block.k_traj = inline_init_traj_corr(block)       # TODO bjs - do I need to save k_data

    xino, xine = inline_init_interp_kx(block)
    expo, expe = inline_init_process_kt(block, reverse=False)

    block.xino = xino
    block.xine = xine
    block.expo = expo
    block.expe = expe

    block.ref_done = True


def process_init_epsi_part(block, group, config, metadata):
    """ Format data into a [cha RO ave lin seg] array """

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = [item.idx.segment for item in group]

    if len(set(indz)) > 1:
        logger_bjs.info("Too many Z encodes in Init data group")
    if len(set(indy)) > 1:
        logger_bjs.info("Too many Y encodes in Init data group")
    if len(set(indt)) != block.nt * 2:
        logger_bjs.info("Length of segment encodes list not equal to Nt in Init data group")

    for item in group:
        if item.idx.contrast == 1:
            for icha in range(block.ncha):
                it = item.idx.segment % block.nt
                block.ref_part[icha, it, :] = item.data[icha,:]
                # NB. bjs, only using FULL ref data to init xino, xine, expo, expe
                #  but will stor this here just in case I want to check

    # # Ref acq should be in block.water/metab[:][0,0,:,:] arrays
    #
    # block.k_traj = inline_init_traj_corr(block)       # TODO bjs - do I need to save k_data
    #
    # xino, xine = inline_init_interp_kx(block)
    # expo, expe = inline_init_process_kt(block, reverse=False)
    #
    # block.xino = xino
    # block.xine = xine
    # block.expo = expo
    # block.expe = expe
    #
    # block.ref_done = True


def process_raw_to_epsi_full(block, group):
    ''' group is one EPI readout of water OR metab '''

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = [item.idx.segment for item in group]
    ieco = group[0].idx.contrast

    if len(set(indz)) > 1:
        logger_bjs.info("process_raw_to_epsi_full(): Too many Z encodes in TR data group")
    if len(set(indy)) > 1:
        logger_bjs.info("process_raw_to_epsi_full(): Too many Y encodes in TR data group")
    if len(set(indt)) != block.nt:
        logger_bjs.info("process_raw_to_epsi_full(): Length of segment encodes list not equal to Nt in Init data group")

    for item in group:
        it = item.idx.segment % block.nt
        for icha in range(block.ncha):
            if item.idx.contrast == 0:
                block.metab_full[icha, it, :] = item.data[icha,:]
            else:
                block.water_full[icha, it, :] = item.data[icha,:]

    if ieco == 0:
        dat_out = inline_do_epsi(block, block.metab_full)
        block.metab_epsi_full[indy, :, :, :] = dat_out  # dims should be ncha, nx2, nt2 here
    else:
        dat_out = inline_do_epsi(block, block.water_full)
        block.water_epsi_full[indy, :, :, :] = dat_out

    return


def process_raw_to_epsi_part(block, group):
    ''' group is one EPI readout of water OR metab '''

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = [item.idx.segment for item in group]
    ieco = group[0].idx.contrast

    if len(set(indz)) > 1:
        logger_bjs.info("process_raw_to_epsi_part(): Too many Z encodes in TR data group")
    if len(set(indy)) > 1:
        logger_bjs.info("process_raw_to_epsi_part(): Too many Y encodes in TR data group")
    if len(set(indt)) != block.nt:
        logger_bjs.info("process_raw_to_epsi_part(): Length of segment encodes list not equal to Nt in Init data group")

    for item in group:
        it = item.idx.segment % block.nt
        for icha in range(block.ncha):
            if item.idx.contrast == 0:
                block.metab_part[icha, it, :] = item.data[icha,:]
            else:
                block.water_part[icha, it, :] = item.data[icha,:]

    if ieco == 0:
        dat_out = inline_do_epsi(block, block.metab_part)
        block.metab_epsi_part[indy, :, :, :] = dat_out  # dims should be ncha, nx2, nt2 here
    else:
        dat_out = inline_do_epsi(block, block.water_part)
        block.water_epsi_part[indy, :, :, :] = dat_out

    return


def send_raw_blank(block, group, metadata, ser_num, zindx, flag_full=False):

    #zindx = block.last_zindx_full
    images = []

    posvecx, posvecy, posvecz = calc_posvec(block, zindx, metadata)

    # Set ISMRMRD Meta Attributes
    tmpMetaMet, tmpMetaWat = default_meta(block, zindx, metadata)
    if flag_full:
        tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-METAB-FULL']
        tmpMetaMet['SequenceDescriptionAdditional'] = 'RAW_FULL'
        tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-WATER-FULL']
        tmpMetaWat['SequenceDescriptionAdditional'] = 'RAW_FULL'
    else:
        tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-METAB-PART']
        tmpMetaMet['SequenceDescriptionAdditional'] = 'RAW_PART'
        tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-WATER-PART']
        tmpMetaWat['SequenceDescriptionAdditional'] = 'RAW_PART'

    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        #metab = block.metab_raw_full[icha][zindx, :,:,:].copy()
        #water = block.water_raw_full[icha][zindx, :,:,:].copy()

        metab = block.blank_slice[zindx, :,:,:].copy()  # every channel the same
        water = block.blank_slice[zindx, :,:,:].copy()

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1], ms[2]
        water.shape = ws[0], ws[1], ws[2]

        # metab = np.conj(metab)
        # water = np.conj(water)

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        tmpImgMet.image_series_index = ser_num
        tmpImgWat.image_series_index = ser_num

        tmpImgMet.image_index = block.out_indx_raw + 0
        tmpImgWat.image_index = block.out_indx_raw + 1

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

        block.out_indx_raw += 2

    return images


def send_epsi_blank(block, group, metadata, ser_num, zindx, flag_full=False):

    #zindx = block.last_zindx_epsi_full
    images = []

    posvecx, posvecy, posvecz = calc_posvec(block, zindx, metadata)

    # Set ISMRMRD Meta Attributes
    tmpMetaMet, tmpMetaWat = default_meta(block, zindx, metadata)

    if flag_full:
        tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-METAB-FULL']
        tmpMetaMet['SequenceDescriptionAdditional'] = 'GRID_FULL'
        tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-WATER-FULL']
        tmpMetaWat['SequenceDescriptionAdditional'] = 'GRID_FULL'
    else:
        tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-METAB-PART']
        tmpMetaMet['SequenceDescriptionAdditional'] = 'GRID_PART'
        tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-WATER-PART']
        tmpMetaWat['SequenceDescriptionAdditional'] = 'GRID_PART'

    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        #metab = block.metab_epsi_full[:, icha, :, :].copy()
        #water = block.water_epsi_full[:, icha, :, :].copy()

        metab = block.blank_epsi_slice[:, icha, :, :].copy()
        water = block.blank_epsi_slice[:, icha, :, :].copy()

        metab = np.squeeze(metab)
        water = np.squeeze(water)

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1], ms[2]
        water.shape = ws[0], ws[1], ws[2]

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        tmpImgMet.image_series_index = ser_num
        tmpImgWat.image_series_index = ser_num

        tmpImgMet.image_index = block.out_indx_epsi + 0
        tmpImgWat.image_index = block.out_indx_epsi + 1

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

        block.out_indx_epsi += 2

    return images


def send_raw_full(block, group, metadata, ser_num):

    zindx = block.last_zindx_full
    images = []

    posvecx, posvecy, posvecz = calc_posvec(block, zindx, metadata)

    # Set ISMRMRD Meta Attributes
    tmpMetaMet, tmpMetaWat = default_meta(block, zindx, metadata)
    tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-METAB-FULL']
    tmpMetaMet['SequenceDescriptionAdditional'] = 'RAW_FULL'
    tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-WATER-FULL']
    tmpMetaWat['SequenceDescriptionAdditional'] = 'RAW_FULL'

    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab_raw_full[icha][zindx, :,:,:].copy()
        water = block.water_raw_full[icha][zindx, :,:,:].copy()

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1], ms[2]
        water.shape = ws[0], ws[1], ws[2]

        # metab = np.conj(metab)
        # water = np.conj(water)

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        tmpImgMet.image_series_index = ser_num
        tmpImgWat.image_series_index = ser_num

        tmpImgMet.image_index = block.out_indx_raw + 0
        tmpImgWat.image_index = block.out_indx_raw + 1

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

        block.out_indx_raw += 2

    return images


def send_raw_part(block, group, metadata, ser_num):

    zindx = block.last_zindx_part
    images = []

    posvecx, posvecy, posvecz = calc_posvec(block, zindx, metadata)

    # Set ISMRMRD Meta Attributes
    tmpMetaMet, tmpMetaWat = default_meta(block, zindx, metadata)
    tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-METAB-PART']
    tmpMetaMet['SequenceDescriptionAdditional'] = 'RAW_PART'
    tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'RAW-WATER-PART']
    tmpMetaWat['SequenceDescriptionAdditional'] = 'RAW_PART'

    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab_raw_part[icha][zindx, :,:,:].copy()
        water = block.water_raw_part[icha][zindx, :,:,:].copy()

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1], ms[2]
        water.shape = ws[0], ws[1], ws[2]

        # metab = np.conj(metab)
        # water = np.conj(water)

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        tmpImgMet.image_series_index = ser_num
        tmpImgWat.image_series_index = ser_num

        tmpImgMet.image_index = block.out_indx_raw + 0
        tmpImgWat.image_index = block.out_indx_raw + 1

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

        block.out_indx_raw += 2

    return images


def send_epsi_full(block, group, metadata, ser_num):

    zindx = block.last_zindx_epsi_full
    images = []

    posvecx, posvecy, posvecz = calc_posvec(block, zindx, metadata)

    # Set ISMRMRD Meta Attributes
    tmpMetaMet, tmpMetaWat = default_meta(block, zindx, metadata)
    tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-METAB-FULL']
    tmpMetaMet['SequenceDescriptionAdditional'] = 'GRID_FULL'
    tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-WATER-FULL']
    tmpMetaWat['SequenceDescriptionAdditional'] = 'GRID_FULL'

    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab_epsi_full[:, icha, :, :].copy()
        water = block.water_epsi_full[:, icha, :, :].copy()

        metab = np.squeeze(metab)
        water = np.squeeze(water)

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1], ms[2]
        water.shape = ws[0], ws[1], ws[2]

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        tmpImgMet.image_series_index = ser_num
        tmpImgWat.image_series_index = ser_num

        tmpImgMet.image_index = block.out_indx_epsi + 0
        tmpImgWat.image_index = block.out_indx_epsi + 1

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

        block.out_indx_epsi += 2

    return images


def send_epsi_part(block, group, metadata, ser_num):

    zindx = block.last_zindx_epsi_part
    images = []

    posvecx, posvecy, posvecz = calc_posvec(block, zindx, metadata)

    # Set ISMRMRD Meta Attributes
    tmpMetaMet, tmpMetaWat = default_meta(block, zindx, metadata)
    tmpMetaMet['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-METAB-PART']
    tmpMetaMet['SequenceDescriptionAdditional'] = 'GRID_PART'
    tmpMetaWat['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON', 'PYMIDAS', 'GRID-WATER-PART']
    tmpMetaWat['SequenceDescriptionAdditional'] = 'GRID_PART'

    xml_metab = tmpMetaMet.serialize()
    xml_water = tmpMetaWat.serialize()
    logging.debug("Image MetaAttributes: %s", xml_metab)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab_epsi_part[:, icha, :, :].copy()
        water = block.water_epsi_part[:, icha, :, :].copy()

        metab = np.squeeze(metab)
        water = np.squeeze(water)

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1], ms[2]
        water.shape = ws[0], ws[1], ws[2]

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        tmpImgMet.image_series_index = ser_num
        tmpImgWat.image_series_index = ser_num

        tmpImgMet.image_index = block.out_indx_epsi + 0
        tmpImgWat.image_index = block.out_indx_epsi + 1

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))
        tmpImgWat.position = (ctypes.c_float(posvecx), ctypes.c_float(posvecy), ctypes.c_float(posvecz))

        tmpImgMet.attribute_string = xml_metab
        tmpImgWat.attribute_string = xml_water

        images.append(tmpImgMet)
        images.append(tmpImgWat)

        block.out_indx_epsi += 2

    return images


def calc_posvec(block, zindx, metadata):

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

    return posvecx, posvecy, posvecz


def default_meta(block, zindx, metadata):

    slthick = float(block.fovz/block.nz)

    metadata.encoding[0].echoTrainLength = 2

    # Set ISMRMRD Meta Attributes
    tmpMetaMet = ismrmrd.Meta()
    tmpMetaMet['DataRole'] = 'Spectroscopy'
    tmpMetaMet['Keep_image_geometry'] = 1
    tmpMetaMet['SiemensControl_SpectroData'] = ['bool', 'true']
    tmpMetaMet['InternalSend'] = 1      # skips SpecSend functor in ICE program - might keep header from modification?

    tmpMetaMet['SiemensDicom_EchoTrainLength'] = ['int', 2]
    tmpMetaMet['SiemensDicom_PercentPhaseFoV'] = ['double', 1.0]
    tmpMetaMet['SiemensDicom_PercentSampling'] = ['double', 1.0]
    tmpMetaMet['SiemensDicom_NoOfCols'] = ['int', int(block.nx)]
    tmpMetaMet['SiemensDicom_SliceThickness'] = ['double', slthick]
    tmpMetaMet['SiemensDicom_ProtocolSliceNumber'] = ['double', zindx]
    tmpMetaMet['SiemensDicom_TE'] = ['double', metadata.sequenceParameters.TE[0]]
    tmpMetaMet['SiemensDicom_TR'] = ['double', metadata.sequenceParameters.TR[0]]
    tmpMetaMet['SiemensDicom_TI'] = ['double', metadata.sequenceParameters.TI[0]]
    tmpMetaMet['SiemensDicom_PixelSpacing'] = [float(block.fovx / block.nx), float(block.fovy / block.ny)]
    tmpMetaMet['SiemensDicom_RealDwellTime'] = ['int', str(int(block.sampling_interval * 1e6 * 1000 * 2))]

    tmpMetaWat = ismrmrd.Meta()
    tmpMetaWat['DataRole'] = 'Spectroscopy'
    tmpMetaWat['Keep_image_geometry'] = 1
    tmpMetaWat['SiemensControl_SpectroData'] = ['bool', 'true']
    tmpMetaWat['InternalSend'] = 1

    tmpMetaWat['SiemensDicom_EchoTrainLength'] = ['int', 2]
    tmpMetaWat['SiemensDicom_PercentPhaseFoV'] = ['double', 1.0]
    tmpMetaWat['SiemensDicom_PercentSampling'] = ['double', 1.0]
    tmpMetaWat['SiemensDicom_NoOfCols'] = ['int', int(block.nx)]
    tmpMetaWat['SiemensDicom_SliceThickness'] = ['double', slthick]
    tmpMetaWat['SiemensDicom_ProtocolSliceNumber'] = ['double', zindx]
    tmpMetaWat['SiemensDicom_TE'] = ['double', metadata.sequenceParameters.TE[1]]
    tmpMetaWat['SiemensDicom_TR'] = ['double', metadata.sequenceParameters.TR[1]]
    tmpMetaWat['SiemensDicom_TI'] = ['double', metadata.sequenceParameters.TI[0]]
    tmpMetaWat['SiemensDicom_PixelSpacing'] = [float(block.fovx / block.nx), float(block.fovy / block.ny)]
    tmpMetaWat['SiemensDicom_RealDwellTime'] = ['int', str(int(block.sampling_interval * 1e6 * 1000 * 2))]

    return tmpMetaMet, tmpMetaWat


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
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1):  lines.append("ACQ_LAST_IN_ENCODE_STEP1               = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2): lines.append("ACQ_FIRST_IN_ENCODE_STEP2              = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2):  lines.append("ACQ_LAST_IN_ENCODE_STEP2               = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_AVERAGE):      lines.append("ACQ_FIRST_IN_AVERAGE                   = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_AVERAGE):       lines.append("ACQ_LAST_IN_AVERAGE                    = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SLICE):        lines.append("ACQ_FIRST_IN_SLICE                     = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):         lines.append("ACQ_LAST_IN_SLICE                      = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_CONTRAST):     lines.append("ACQ_FIRST_IN_CONTRAST                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_CONTRAST):      lines.append("ACQ_LAST_IN_CONTRAST                   = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_PHASE):        lines.append("ACQ_FIRST_IN_PHASE                     = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_PHASE):         lines.append("ACQ_LAST_IN_PHASE                      = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_REPETITION):   lines.append("ACQ_FIRST_IN_REPETITION                = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):    lines.append("ACQ_LAST_IN_REPETITION                 = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SET):          lines.append("ACQ_FIRST_IN_SET                       = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SET):           lines.append("ACQ_LAST_IN_SET                        = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SEGMENT):      lines.append("ACQ_FIRST_IN_SEGMENT                   = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SEGMENT):       lines.append("ACQ_LAST_IN_SEGMENT                    = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):  lines.append("ACQ_IS_NOISE_MEASUREMENT               = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION): lines.append("ACQ_IS_PARALLEL_CALIBRATION          = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING): lines.append("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING= True")
    if item.is_flag_set(ismrmrd.ACQ_IS_REVERSE):            lines.append("ACQ_IS_REVERSE                         = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):    lines.append("ACQ_IS_NAVIGATION_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):     lines.append("ACQ_IS_PHASECORR_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):   lines.append("ACQ_LAST_IN_MEASUREMENT                = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_HPFEEDBACK_DATA):    lines.append("ACQ_IS_HPFEEDBACK_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):     lines.append("ACQ_IS_DUMMYSCAN_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_RTFEEDBACK_DATA):    lines.append("ACQ_IS_RTFEEDBACK_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA): lines.append("ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE):lines.append("ACQ_IS_PHASE_STABILIZATION_REFERENCE   = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION):lines.append("ACQ_IS_PHASE_STABILIZATION             = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION1):          lines.append("ACQ_COMPRESSION1                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION2):          lines.append("ACQ_COMPRESSION2                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION3):          lines.append("ACQ_COMPRESSION3                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION4):          lines.append("ACQ_COMPRESSION4                       = True")
    if item.is_flag_set(ismrmrd.ACQ_USER1):                 lines.append("ACQ_USER1                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER2):                 lines.append("ACQ_USER2                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER3):                 lines.append("ACQ_USER3                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER4):                 lines.append("ACQ_USER4                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER5):                 lines.append("ACQ_USER5                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER6):                 lines.append("ACQ_USER6                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER7):                 lines.append("ACQ_USER7                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER8):                 lines.append("ACQ_USER8                              = True")

    if lines == []:
        lines = 'No active flags.'
    else:
        lines = "\n".join(lines)

    if prnt == True:
        print(lines)
    else:
        return lines

