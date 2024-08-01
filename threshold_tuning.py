import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *


def get_metrics(dfs):
    note_p, note_r, note_f1 = [], [], []
    frame_p, frame_r, frame_f1 = [], [], []

    for df in dfs:
        note_p.append(df.loc[df['filename'] == 'filewise_mean', ['metric/note/precision']].values.squeeze().item())
        note_r.append(df.loc[df['filename'] == 'filewise_mean', ['metric/note/recall']].values.squeeze().item())
        note_f1.append(df.loc[df['filename'] == 'filewise_mean', ['metric/note/f1']].values.squeeze().item())

        frame_p.append(df.loc[df['filename'] == 'filewise_mean', ['metric/frame/precision']].values.squeeze().item())
        frame_r.append(df.loc[df['filename'] == 'filewise_mean', ['metric/frame/recall']].values.squeeze().item())
        frame_f1.append(df.loc[df['filename'] == 'filewise_mean', ['metric/frame/f1']].values.squeeze().item())

    metrics = {
        'note_p': np.array(note_p),
        'note_r': np.array(note_r),
        'note_f1': np.array(note_f1),
        'frame_p': np.array(frame_p),
        'frame_r': np.array(frame_r),
        'frame_f1': np.array(frame_f1)
    } 

    return metrics


def create_df(dfs_onset, dfs_frame, thresholds, onset_th_opt):
    rows_onset = [df.loc[df['filename'] == 'filewise_mean'] for df in dfs_onset]
    df_onset = pd.concat(rows_onset, axis=0).reset_index(drop=True)
    df_onset.insert(0, 'onset_th', thresholds)
    df_onset.insert(1, 'frame_th', 0.5)

    rows_frame = [df.loc[df['filename'] == 'filewise_mean'] for df in dfs_frame]
    df_frame = pd.concat(rows_frame, axis=0).reset_index(drop=True)
    df_frame.insert(0, 'onset_th', onset_th_opt)
    df_frame.insert(1, 'frame_th', thresholds)

    df = pd.concat([df_onset, df_frame], axis=0).reset_index(drop=True)
    return df


def tune_thresholds(logdir, model, dataset, dataset_id='', thresholds=np.linspace(0, 1, num=101), rng_seed=42):
    ### onset-th evaluation
    dfs_onset = []
    
    for onset_th in tqdm(thresholds, desc='Onset threshold tuning'):
        if rng_seed is not None:
            # reset dataset for getting identical segments for every th
            for ds in dataset.datasets:
                ds.random = np.random.RandomState(rng_seed)

        df = pd.DataFrame([])
        
        with torch.no_grad():
            # frame th does not affect note F1; just use 0.5 here
            metrics, filenames = evaluate(dataset, model, onset_th, 0.5, save_path=None, return_filenames=True)
            
        metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
        metrics_std = {k: np.std(v) for k, v in metrics.items()}
        
        metrics['filename'] = filenames
        metrics_mean['filename'] = 'filewise_mean'
        metrics_std['filename'] = 'filewise_std'
        
        df = pd.concat([pd.DataFrame(metrics), pd.DataFrame(metrics_mean, index=[0]), pd.DataFrame(metrics_std, index=[0])], ignore_index=True)
        dfs_onset.append(df)

    metrics_onset_th = get_metrics(dfs_onset)

    # select onset th that maximizes note F1
    onset_th_opt = thresholds[metrics_onset_th['note_f1'].argmax()]

    ### frame-th evaluation
    dfs_frame = []
    
    for frame_th in tqdm(thresholds, desc='Frame threshold tuning'):
        if rng_seed is not None:
            # reset dataset for getting identical segments for every th
            for ds in dataset.datasets:
                ds.random = np.random.RandomState(rng_seed)   
            
        df = pd.DataFrame([])
        
        with torch.no_grad():
            metrics, filenames = evaluate(dataset, model, onset_th_opt, frame_th, save_path=None, return_filenames=True)
            
        metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
        metrics_std = {k: np.std(v) for k, v in metrics.items()}
        
        metrics['filename'] = filenames
        metrics_mean['filename'] = 'filewise_mean'
        metrics_std['filename'] = 'filewise_std'
        
        df = pd.concat([pd.DataFrame(metrics), pd.DataFrame(metrics_mean, index=[0]), pd.DataFrame(metrics_std, index=[0])], ignore_index=True)
        dfs_frame.append(df)

    metrics_frame_th = get_metrics(dfs_frame)

    frame_th_opt = thresholds[metrics_frame_th['frame_f1'].argmax()]

    df = create_df(dfs_onset, dfs_frame, thresholds, onset_th_opt)
    df.to_csv(os.path.join(logdir, f'threshold_tuning_{dataset_id}.csv'), index=False)

    th_opt = {
        'onset_th': onset_th_opt,
        'frame_th': frame_th_opt
    }

    df_opt = pd.DataFrame(th_opt, index=[0])
    df_opt.to_csv(os.path.join(logdir, f'thresholds_opt.csv'), index=False)

    return onset_th_opt, frame_th_opt
