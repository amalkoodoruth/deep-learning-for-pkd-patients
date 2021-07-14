import shutil
import os

master_path = '/Users/amalnabeelkoodoruth/git/deep-learning-for-pkd-patients'
master_dir = master_path + '/MR/'

for patient in os.listdir(master_dir):
    if patient != '.DS_Store':
        dcm_dir = master_dir + patient + '/T2SPIR/DICOM_anon/'
        seg_dir = master_dir + patient + '/T2SPIR/Ground/'

        for dcm in os.listdir(dcm_dir):
            if dcm != '.DS_Store':
                dcm_path = dcm_dir + dcm
                new_dcm_path = master_path + '/beta/MR2D/Train/X/' + patient + '-' + dcm
                shutil.copyfile(dcm_path,new_dcm_path)

        for seg in os.listdir(seg_dir):
            if seg != '.DS_Store':
                seg_path = seg_dir + seg
                new_seg_path = master_path + '/beta/MR2D/Train/Y/' + patient + '-' + seg
                shutil.copyfile(seg_path, new_seg_path)

