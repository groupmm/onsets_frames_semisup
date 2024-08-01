import os
from datetime import datetime
import shutil
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchinfo
from tqdm import tqdm

from evaluate import evaluate
from threshold_tuning import tune_thresholds
from onsets_and_frames import *

ex = Experiment('train_transcriber')


@ex.config
def config():
    resume_from = None
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    if resume_from is not None:
        shutil.copytree(resume_from, logdir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    reset_optimizer = False
    checkpoint_interval = 10000

    # seed = 446187021

    train_on_l = ['MAPS']
    train_on_u = ['MAPS']
    validate_on = ['MAPS']
    test_on = ['MAPS', 'MAESTRO', 'SMD']

    maps_config = {
        'groups_l': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
        'groups_u': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
        'groups_val': ['SptkBGAm'],
        'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
        'avoid_train_test_overlap': True,
        'single_fn_l': False,   # 'chp_op31'
    }

    maestro_config = {
        'groups_l': ['train'],
        'groups_u': ['train'],
        'groups_val': ['validation'],
        'groups_test': ['test'],
    }

    strong_aug_config = {
        'noise_std': 0.01, 
        'n_bands': 30
    }

    load_optimum_th = False
    
    batch_size_l = 8
    batch_size_u = 8
    sequence_length = 327680
    model_complexity = 48
    learning_rate = 0.00006
    learning_rate_decay_steps = 5000
    learning_rate_decay_rate = 0.98
    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    test_length = None

    lambda_u = 0.05
    unsupervised_start_it = None

    pl_th_lower = 0.05
    pl_th_upper = 0.95

    pl_loss_weights = {'onset': 1., 'offset': 0., 'frame': 1., 'velocity': 0.}

    strong_aug_l = False
    strong_aug_u = True

    distribution_matching = True   

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, reset_optimizer, checkpoint_interval, 
          train_on_l, train_on_u, validate_on, test_on, maps_config, maestro_config, strong_aug_config, load_optimum_th, 
          batch_size_l, batch_size_u, sequence_length, model_complexity, learning_rate, learning_rate_decay_steps, 
          learning_rate_decay_rate, clip_gradient_norm, validation_length, validation_interval, test_length, 
          lambda_u, unsupervised_start_it, pl_th_lower, pl_th_upper, pl_loss_weights, 
          strong_aug_l, strong_aug_u, distribution_matching):
    
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    ### Prepare Datasets ##################################
    # labeled training dataset
    dataset_l = []
    for ds in train_on_l:
        if ds == 'MAESTRO':
            dataset_l.append(MAESTRO(groups=maestro_config['groups_l'], sequence_length=sequence_length, device=device, labeled=True))
        elif ds == 'MAPS':
            dataset_l.append(MAPS(groups=maps_config['groups_l'], sequence_length=sequence_length, device=device, labeled=True, single_fn=maps_config['single_fn_l'], avoid_train_test_overlap=maps_config['avoid_train_test_overlap']))
        elif ds == 'SMD':
            dataset_l.append(SMD(groups=['SMD'], sequence_length=sequence_length, device=device, labeled=True))
    dataset_l = torch.utils.data.ConcatDataset(dataset_l)

    # unlabeled training dataset
    dataset_u = []
    for ds in train_on_u:
        if ds == 'MAESTRO':
            dataset_u.append(MAESTRO(groups=maestro_config['groups_u'], sequence_length=sequence_length, device=device, labeled=False))
        elif ds == 'MAPS':
            dataset_u.append(MAPS(groups=maps_config['groups_u'], sequence_length=sequence_length, device=device, labeled=False, avoid_train_test_overlap=maps_config['avoid_train_test_overlap']))
        elif ds == 'SMD':
            dataset_u.append(SMD(groups=['SMD'], sequence_length=sequence_length, device=device, labeled=False))
    dataset_u = torch.utils.data.ConcatDataset(dataset_u)

    # validation dataset
    validation_dataset = []
    for ds in validate_on:
        if ds == 'MAESTRO':
            validation_dataset.append(MAESTRO(groups=maestro_config['groups_val'], sequence_length=sequence_length))
        elif ds == 'MAPS':
            validation_dataset.append(MAPS(groups=maps_config['groups_val'], sequence_length=validation_length))
        elif ds == 'SMD':
            validation_dataset.append(SMD(groups=['SMD'], sequence_length=validation_length))
    validation_dataset = torch.utils.data.ConcatDataset(validation_dataset)

    # dataloaders
    loader_l = DataLoader(dataset_l, batch_size_l, shuffle=True, drop_last=True)
    loader_u = DataLoader(dataset_u, batch_size_u, shuffle=True, drop_last=True)

    # correct batch size if necessary
    batch_size_l = min(batch_size_l, len(loader_l))
    batch_size_u = min(batch_size_u, len(loader_u))

    # determine class frequencies in training set 
    for i in range(len(dataset_l.datasets)):
        dataset_l.datasets[i].sequence_length = None        # get whole piece
        
    class_occurrences = {'onset': 0, 'offset': 0, 'frame': 0}
    n_tf_bins = 0

    for batch in dataset_l:
        n_tf_bins += batch['onset'].numel()

        for k in ['onset', 'offset', 'frame']:
            class_occurrences[k] += batch[k].sum().item()
    
    class_frequencies = {k: v / n_tf_bins for k, v in class_occurrences.items()}

    for i in range(len(dataset_l.datasets)):
        dataset_l.datasets[i].sequence_length = sequence_length

    ### Create Model & Optimizer ##########################
    if resume_iteration is None:
        model_params = {'input_features': N_MELS, 
                        'output_features': MAX_MIDI - MIN_MIDI + 1,
                        'model_complexity': model_complexity}
        
        model = OnsetsAndFrames(**model_params).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        if not reset_optimizer:
            optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    # summary(model)
    torchinfo.summary(model, depth=4, input_size=(1, 640, 229))
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    ### Load / Set Thresholds #############################     
    if load_optimum_th:
        th_file = os.path.join(logdir, f'thresholds_opt.csv')
        assert os.path.isfile(th_file), 'Can not find file that contains optimum thresholds...'
        th_df = pd.read_csv(th_file)
        th_onset, th_frame = th_df.loc[0, 'onset_th'], th_df.loc[0, 'frame_th']
    else:
        th_onset, th_frame = 0.5, 0.5

    ### Training Loop #####################################
    loop = tqdm(range(resume_iteration + 1, iterations + 1), desc='Training')
    for i, batch_l, batch_u in zip(loop, cycle(loader_l), cycle(loader_u)):
        # labeled data
        if strong_aug_l:
            mel = melspectrogram(batch_l['audio'].reshape(-1, batch_l['audio'].shape[-1])[:, :-1]).transpose(-1, -2)
            mel = apply_strong_aug(mel, **strong_aug_config)
        else:
            mel = None

        _, losses_l = model.run_on_batch(batch_l, labeled=True, mel=mel)

        # unlabeled data
        if (unsupervised_start_it is not None) and (i > unsupervised_start_it):
            # get predictions (clean data, no gradients)
            with torch.no_grad():
                predictions_u, _ = model.run_on_batch(batch_u, labeled=False, pseudo_labels=None)

            # convert predictions into pseudo-labels
            pseudo_labels_u = get_pseudo_labels(predictions_u, th_lower=pl_th_lower, th_upper=pl_th_upper, class_frequencies=class_frequencies, 
                                                distribution_matching=distribution_matching, ignore_index=-1, writer=writer, global_step=i)

            # apply strong augmentation if necessary
            if strong_aug_u:
                mel = melspectrogram(batch_u['audio'].reshape(-1, batch_u['audio'].shape[-1])[:, :-1]).transpose(-1, -2)
                mel = apply_strong_aug(mel, **strong_aug_config)
            else:
                mel = None

            # get predictions ((noisy) data, with gradients)
            _, losses_u = model.run_on_batch(batch_u, labeled=False, pseudo_labels=pseudo_labels_u, loss_weights=pl_loss_weights, mel=mel)

            # unsupervised loss scaling
            losses_u = {k: v * lambda_u for k, v in losses_u.items()}
        else:
            losses_u = {}
            lambda_u = 0.0

        losses_l = {k: v * (1 - lambda_u) for k, v in losses_l.items()}

        losses = {}
        losses.update({k.replace('loss', 'loss_l'): v for k, v in losses_l.items()})
        losses.update({k.replace('loss', 'loss_u'): v for k, v in losses_u.items()})

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
        optimizer.step()
        scheduler.step()        

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model, th_onset, th_frame).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

    ### Threshold Tuning ##################################
    model.eval()
    th_onset, th_frame = tune_thresholds(logdir=logdir, model=model, dataset=validation_dataset, dataset_id='_'.join(validate_on), thresholds=np.linspace(0, 1, num=101))

    ### Testing ###########################################
    for curr_test_ds in test_on:
        # test dataset
        if curr_test_ds == 'MAESTRO':
            test_dataset = MAESTRO(groups=maestro_config['groups_test'], sequence_length=test_length)
        elif curr_test_ds == 'MAPS':
            test_dataset = MAPS(groups=maps_config['groups_test'], sequence_length=test_length, avoid_train_test_overlap=False)
        elif curr_test_ds == 'SMD':
            test_dataset = SMD(groups=['SMD'], sequence_length=test_length)

        df = pd.DataFrame([])

        with torch.no_grad():
            metrics, filenames = evaluate(tqdm(test_dataset, desc=f'Testing on {curr_test_ds}'), model, th_onset, th_frame, save_path=None, return_filenames=True)
            
        metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
        metrics_std = {k: np.std(v) for k, v in metrics.items()}

        metrics['filename'] = filenames
        metrics_mean['filename'] = 'filewise_mean'
        metrics_std['filename'] = 'filewise_std'

        df = pd.concat([pd.DataFrame(metrics), pd.DataFrame(metrics_mean, index=[0]), pd.DataFrame(metrics_std, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(logdir, f'test_results_{curr_test_ds}.csv'))
