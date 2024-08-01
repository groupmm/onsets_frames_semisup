# Semi-Supervised Piano Transcription Using Pseudo-Labeling Techniques

This is a Pytorch code repository accompanying the following paper:  

```bibtex
@inproceedings{StrahlM24_SemiSupPianoTranscription_ISMIR,
  author    = {Sebastian Strahl and Meinard M{\"u}ller},
  title     = {Semi-Supervised Piano Transcription Using Pseudo-Labeling Techniques},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  address   = {San Francisco, USA},
  year      = {2024}
}
```

This repository contains code for all of the paper's experiments. 
The codebase builds upon the [PyTorch Implementation of Onsets and Frames by Jong Wook Kim](https://github.com/jongwook/onsets-and-frames).

All datasets used in the paper are publicly available:
- [MAPS (MIDI Aligned Piano Sounds)](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/)
- [MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization)](https://magenta.tensorflow.org/datasets/maestro)
- [SMD (Saarland Music Data: MIDI-Audio Piano Music)](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi)

For details and references, please see the paper.

# Instructions
## Installation
```bash
cd onsets_frames_semisup
conda env create -f environment.yml
conda activate onsets_frames_semisup
```

## Data Preparation
With the following steps, the required datasets can be downloaded and prepared. All audio data is resampled to 16 kHz, and the data which is not needed for the experiments will be deleted. Note that the data preparation requires about 200GB space for intermediate storage and that ```ffmpeg``` needs to be installed. 

1. Download the pre-processed MAPS dataset from [here](https://github.com/jongwook/onsets-and-frames/tree/master/data/MAPS) by running

    ```bash
    cd data
    ./download_maps.sh
    ```

2. Determine the training set pieces of the MAPS datasets which do not overlap with the test set by running

    ```bash
    python get_MAPS_train_test_overlap.py
    ```

3. Download and prepare the MAESTRO V3.0.0 dataset from [here](https://magenta.tensorflow.org/datasets/maestro) by running 

    ```bash
    ./prepare_maestro.sh

    ```

4. Download the SMD dataset from [here](https://zenodo.org/records/10847281) , convert the annotations into tsv files, resample to 16 kHz, and delete data which is not used by running

    ```bash
    ./download_smd.sh
    python prepare_smd.py
    cd SMD 
    rm -rf csv midi midi_wav_22050_mono wav_22050_mono wav_44100_stereo
    cd ../..
    ```


## Experiments
To reproduce all the results from the paper, run the following command: \
(Note that carrying out these experiments requires at least 20GB GPU memory.)
```bash
python run_experiments.py
```
Trained models and testing results are stored in ```runs```.

# Acknowledgements:
This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 350953655 (MU 2686/11-2) and Grant No. 500643750 (MU 2686/15-1). The authors are with the [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/), a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for 
Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).