import torch

def get_pseudo_labels(predictions, th_lower=0.05, th_upper=0.95, class_frequencies=None, 
                      distribution_matching=False, ignore_index=-1, writer=None, global_step=None):
    """
    This function converts model predictions into pseudo-labels.

    Parameters
    ----------
    predictions: Dict([str, torch.Tensor])
        Model predictions of shape (batch size, frames, bins)

    th_lower: float 
        Lower pseudo-label threshold; all values below will be converted to zeros

    th_upper: float 
        Upper pseudo-label threshold; all values above will be converted to ones

    class_frequencies: Dict([str, float]) 
        Fraction of ones in the (binary) reference labels for onsets, offsets, and frame activity

    distribution_matching: bool
        Whether to perform distribution matching to pseudo-labels (apply undersampling to over-represented class)

    ignore_index: int
        Value to be assigned to unreliable predictions (above lower and below upper threshold)

    writer: torch.utils.tensorboard.SummaryWriter
        Tensorboard writer

    global_step: int
        Current training iteration index

    Returns
    -------
    pseudo_labels: Dict([str, torch.Tensor])
        Pseudo-labels of shape (batch size, frames, bins)
    """        

    ### Get Pseudo-Labels #######################
    pseudo_labels = {}
    keys = ['onset', 'offset', 'frame']
    for k in keys:
        mask_lowconf = torch.zeros_like(predictions[k])
        mask_lowconf[torch.logical_and(predictions[k] > th_lower, predictions[k] < th_upper)] = 1.0

        pseudo_labels[k] = predictions[k].clone()
        pseudo_labels[k][predictions[k] <= th_lower] = 0.0
        pseudo_labels[k][predictions[k] >= th_upper] = 1.0
        pseudo_labels[k][mask_lowconf.type(torch.bool)] = ignore_index

        if distribution_matching:
            n_ones = (pseudo_labels[k] == 1.0).sum().item()
            n_zeros = (pseudo_labels[k] == 0.0).sum().item()

            if (n_zeros > 0) and (n_ones / n_zeros < class_frequencies[k]):   
                 # drop some zeros
                n_zeros_target = int(n_ones / class_frequencies[k])
                mask_zeros = (pseudo_labels[k] == 0.0)
                idx_all = mask_zeros.nonzero(as_tuple=True)
                idx_rand = torch.randperm(n_zeros)[:(n_zeros - n_zeros_target)]
                idx_to_convert = tuple([i[idx_rand] for i in idx_all])
                pseudo_labels[k][idx_to_convert] = ignore_index
            else:     
                # drop some ones
                n_ones_target = int(n_zeros * class_frequencies[k])
                mask_ones = (pseudo_labels[k] == 1.0)
                idx_all = mask_ones.nonzero(as_tuple=True)
                idx_rand = torch.randperm(n_ones)[:(n_ones - n_ones_target)]
                idx_to_convert = tuple([i[idx_rand] for i in idx_all])
                pseudo_labels[k][idx_to_convert] = ignore_index

    pseudo_labels['velocity'] = predictions['velocity'].clone()

    ### Logging #################################
    if writer is not None and global_step is not None:
        for k in ['onset', 'offset', 'frame']:
            # log pseudo label statistics
            writer.add_scalar(f'pl_statistics/{k}/fraction_ones', (pseudo_labels[k] == 1.0).sum().item() / pseudo_labels[k].numel(), global_step=global_step)
            writer.add_scalar(f'pl_statistics/{k}/fraction_zeros', (pseudo_labels[k] == 0.0).sum().item() / pseudo_labels[k].numel(), global_step=global_step)

    return pseudo_labels
