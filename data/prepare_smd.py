import os
import sys
from glob import glob

import math
import csv
from scipy.io import wavfile
import librosa
import numpy as np
from tqdm import tqdm

DIR_SMD = os.path.join(os.getcwd(), 'SMD')
DIR_CSV = os.path.join(DIR_SMD, 'csv')
DIR_TSV = os.path.join(DIR_SMD, 'note_tsv')

DIR_WAV_22050 = os.path.join(DIR_SMD, 'wav_22050_mono')
DIR_WAV_16000 = os.path.join(DIR_SMD, 'wav_16000_mono')

def convert_csv_to_tsv(input_csv, output_tsv):
    with open(input_csv, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        
        # Skip the header row
        next(reader)

        with open(output_tsv, 'w', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')

            # Write the header to the TSV file
            writer.writerow(["# onset,offset,note,velocity"])

            # Convert and write each row to the TSV file
            for row in reader:
                onset = round(float(row[0]), 6)
                offset = round(onset + float(row[1]), 6)
                note = float(row[2])
                velocity = float(math.floor(float(row[3]) * 128))

                writer.writerow([f'{onset:.6f}', f'{offset:.6f}', f'{note:.6f}', f'{velocity:.6f}'])


if __name__ == '__main__':
    # convert .csv into .tsv
    os.makedirs(DIR_TSV, exist_ok=True)

    for file_path_csv in glob(os.path.join(DIR_CSV, '*.csv')):
        fn = os.path.basename(file_path_csv)
        file_path_tsv = os.path.join(DIR_TSV, f'{fn[:-4]}.tsv')
        convert_csv_to_tsv(file_path_csv, file_path_tsv)
    
    # resample audio to 16kHz
    os.makedirs(DIR_WAV_16000, exist_ok=True)

    for file_path_wav in tqdm(glob(os.path.join(DIR_WAV_22050, '*.wav'))):
        audioIn, FsIn = librosa.load(file_path_wav, sr=16000, mono=True)    
        audioOut = audioIn / np.max(np.abs(audioIn))    
        audioOut *= np.iinfo(np.int16).max    
        wavfile.write(filename=os.path.join(DIR_WAV_16000, os.path.basename(file_path_wav)), rate=16000, data=audioOut.astype(np.int16)) 
