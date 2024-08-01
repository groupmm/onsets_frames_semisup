import torch
import torchaudio.functional as taf

def apply_strong_aug(mel, noise_std=0.1, n_bands=30):
    """
    This function applies strong augmentation to mel spectrograms. 
    The used augmentations are frequency masking and addition of Gaussian noise.

    Parameters
    ----------
    mel: torch.Tensor 
        Mel spectrograms of shape (batch size, frames, bins)

    noise_std: float
        Standard deviation of the Gaussian noise to be added

    n_bands: int
        Maximum number of contiguous frequency bins to be masked

    Returns
    -------
    mel: torch.Tensor
        Augmented mel spectrograms of shape (batch size, frames, bins)
    """

    mel = taf.mask_along_axis_iid(mel, n_bands, mel.mean(), 2)
    mel = mel + torch.normal(mean=torch.zeros(mel.size()), std=noise_std * torch.ones(mel.size())).to(mel.device)
    return mel
