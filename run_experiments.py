import os
import importlib
import train_semisup
import warnings
warnings.filterwarnings('ignore')

# select GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### base configuration ####################################

base_config = {
    'resume_from': None,

    # 'logdir': 'runs/xyz',
    'device': 'cuda',
    'iterations': 50000,
    'resume_iteration': None,
    'reset_optimizer': False,
    'checkpoint_interval': 10000,

    # 'seed': 446187021,

    'train_on_l': ['MAPS'],
    'train_on_u': ['MAESTRO'],
    'validate_on': ['MAPS'],
    'test_on': ['MAPS', 'MAESTRO', 'SMD'],

    'maps_config': {
        'groups_l': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
        'groups_u': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
        'groups_val': ['SptkBGAm'],
        'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
        'avoid_train_test_overlap': True,
        'single_fn_l': False,   # 'chp_op31'
    },

    'maestro_config': {
        'groups_l': ['train'],
        'groups_u': ['train'],
        'groups_val': ['validation'],
        'groups_test': ['test'],
    },

    'strong_aug_config': {
        'noise_std': 0.01, 
        'n_bands': 30
    },

    'load_optimum_th': False,
    
    'batch_size_l': 8,
    'batch_size_u': 8,
    'sequence_length': 327680,
    'model_complexity': 48,
    'learning_rate': 0.00006,
    'learning_rate_decay_steps': 5000,
    'learning_rate_decay_rate': 0.98,
    'clip_gradient_norm': 3,

    'validation_length': 327680,
    'validation_interval': 500,

    'test_length': None,

    'lambda_u': 0.05,
    'unsupervised_start_it': None,

    'pl_th_lower': 0.05,
    'pl_th_upper': 0.95,

    'pl_loss_weights': {'onset': 1., 'offset': 0., 'frame': 1., 'velocity': 0.},

    'strong_aug_l': False,
    'strong_aug_u': True,

    'distribution_matching': True,
}


### individual configurations #############################

exps = [

    ### base

    {
        'logdir': 'runs/transcriber-MAPS_all_groups',
        'maps_config': {
            'groups_l': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
    },

    {
        'logdir': 'runs/transcriber-MAPS_one_group',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
    },

    {
        'logdir': 'runs/transcriber-MAPS_one_piece',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': 'chp_op31',
        },
        'batch_size_l': 1,
    },


    ### MAPS (continued) ###

    {
        'resume_from': 'runs/transcriber-MAPS_all_groups',
        'logdir': 'runs/transcriber-MAPS_all_groups-OF',
        'maps_config': {
            'groups_l': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 1000,
        'iterations': 2000,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-OF',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_piece',
        'logdir': 'runs/transcriber-MAPS_one_piece-OF',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': 'chp_op31',
        },
        'batch_size_l': 1,
        'resume_iteration': 50000,
        'iterations': 100000,
    },


    ### semisup MAESTRO ###

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS1',     
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
        'strong_aug_u': False,
        'distribution_matching': False,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS2',   
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
        'strong_aug_u': False,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS3',   
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
        'distribution_matching': False,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS4',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS5',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
        'pl_th_lower': 0.25,
        'pl_th_upper': 0.75,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS6',     
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
        'pl_loss_weights': {'onset': 1., 'offset': 1., 'frame': 1., 'velocity': 0.},
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_group',
        'logdir': 'runs/transcriber-MAPS_one_group-MAESTRO-OFSS7',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
        'lambda_u': 0.01,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_all_groups',
        'logdir': 'runs/transcriber-MAPS_all_groups-MAESTRO-OFSS4',
        'maps_config': {
            'groups_l': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': False,
        },
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
    },

    {
        'resume_from': 'runs/transcriber-MAPS_one_piece',
        'logdir': 'runs/transcriber-MAPS_one_piece-MAESTRO-OFSS4',
        'maps_config': {
            'groups_l': ['AkPnBcht'],
            'groups_val': ['SptkBGAm'],
            'groups_test': ['ENSTDkAm', 'ENSTDkCl'],
            'avoid_train_test_overlap': True,
            'single_fn_l': 'chp_op31',
        },
        'batch_size_l': 1,
        'resume_iteration': 50000,
        'iterations': 100000,
        'unsupervised_start_it': 50000,
    },
]


for exp in exps:
    importlib.reload(train_semisup)
    config_exp = base_config.copy()
    config_exp.update(exp)

    train_semisup.ex.run(config_updates=config_exp)
