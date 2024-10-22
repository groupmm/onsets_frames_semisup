import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, labeled=True):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.labeled = labeled

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)

            if self.labeled:
                result['label'] = data['label'][step_begin:step_end, :].to(self.device)
                result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)

            if self.labeled:
                result['label'] = data['label'].to(self.device)
                result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)

        if self.labeled:
            result['onset'] = (result['label'] == 3).float()
            result['offset'] = (result['label'] == 1).float()
            result['frame'] = (result['label'] > 1).float()
            result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path_audio = f'{audio_path.replace(".flac", "_flac").replace(".wav", "_wav")}.pt'
        saved_data_path_labels = f'{tsv_path.replace(".tsv", "_tsv")}.pt'

        # load audio
        if os.path.exists(saved_data_path_audio):
            data_audio = torch.load(saved_data_path_audio)
        else:
            audio, sr = soundfile.read(audio_path, dtype='int16')
            assert sr == SAMPLE_RATE

            audio = torch.ShortTensor(audio)
            
            data_audio = dict(path=audio_path, audio=audio)
            torch.save(data_audio, saved_data_path_audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (len(data_audio['audio']) - 1) // HOP_LENGTH + 1

        # load labels
        if self.labeled:
            if os.path.exists(saved_data_path_labels):
                data_labels = torch.load(saved_data_path_labels)

            else:
                label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
                velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

                midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

                for onset, offset, note, vel in midi:
                    left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                    onset_right = min(n_steps, left + HOPS_IN_ONSET)
                    frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                    frame_right = min(n_steps, frame_right)
                    offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                    f = int(note) - MIN_MIDI
                    label[left:onset_right, f] = 3
                    label[onset_right:frame_right, f] = 2
                    label[frame_right:offset_right, f] = 1
                    velocity[left:frame_right, f] = vel

                data_labels = dict(label=label, velocity=velocity)
                torch.save(data_labels, saved_data_path_labels)  

            return {**data_audio, **data_labels}

        else:
            return data_audio

class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=['train'], **kwargs):
        super().__init__(path=path, groups=groups, **kwargs)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = pd.read_csv(os.path.join(self.path, 'maestro-v3.0.0.csv'))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for (_, row) in metadata.iterrows() if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=['ENSTDkAm', 'ENSTDkCl'], single_fn=False, avoid_train_test_overlap=False, **kwargs):
        self.single_fn = single_fn
        self.avoid_train_test_overlap = avoid_train_test_overlap
        super().__init__(path=path, groups=groups, **kwargs)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))

        if self.avoid_train_test_overlap:
            train_fns_nooverl = pd.read_csv(os.path.join(self.path, 'train_fns_nooverl.csv'), header=None)[0].values.tolist()
            flacs = [flac for flac in flacs if os.path.basename(flac) in train_fns_nooverl]          

        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        if self.single_fn:
            return [(f, t) for f, t in zip(flacs, tsvs) if self.single_fn in f]
        else:
            return sorted(zip(flacs, tsvs))

class SMD(PianoRollAudioDataset):
    def __init__(self, path='data/SMD', groups=['SMD'], **kwargs):
        super().__init__(path=path, groups=groups, **kwargs)

    @classmethod
    def available_groups(cls):
        return ['SMD']

    def files(self, group):
        wavs = glob(os.path.join(self.path, 'wav_16000_mono', f'*-{group}.wav'))
        tsvs = [f.replace('/wav_16000_mono/', '/note_tsv/').replace('.wav', '.tsv') for f in wavs]

        assert(all(os.path.isfile(wav) for wav in wavs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(wavs, tsvs))
