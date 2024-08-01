import os
from glob import glob
import csv


DATA_DIR = os.path.join(os.getcwd(), 'MAPS')
TEST_GROUPS = ['ENSTDkAm', 'ENSTDkCl']
TRAIN_GROUPS = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']


def get_piece(fn):
    """
     Given a filename, this function returns the ID of the piece
     """
    return ''.join(fn.split('_')[1:-1])

def get_pieces(data_dir, groups):
    """
    Given a list of groups, this function returns the unique pieces in these groups
    """
    test_pieces = []

    for group in groups:
        flacs = glob(os.path.join(data_dir, 'flac', '*_%s.flac' % group))   
        
        for flac in flacs:
            fn = os.path.basename(flac)
            test_pieces.append(get_piece(fn))

    return list(set(test_pieces))

def get_fns_nooverl(data_dir, train_groups, test_pieces):
    """ 
    Given a list of training groups and a list of test pieces, this function returns all
    filenames associated with the training groups that are different from the test pieces
    """
    test_fns = []

    for group in train_groups:
        flacs = glob(os.path.join(data_dir, 'flac', '*_%s.flac' % group))   

        for flac in flacs:
            fn = os.path.basename(flac)
            piece = get_piece(fn)
            if piece not in test_pieces:
                test_fns.append(fn)

    return test_fns
            
        
if __name__ == '__main__':
    test_pieces = get_pieces(DATA_DIR, TEST_GROUPS)
    train_fns_nooverl = get_fns_nooverl(DATA_DIR, TRAIN_GROUPS, test_pieces)

    # create csv file that contains all the filenames of the non-overlapping training set
    with open(os.path.join(DATA_DIR, 'train_fns_nooverl.csv'), 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows([[fn] for fn in sorted(train_fns_nooverl)])   
