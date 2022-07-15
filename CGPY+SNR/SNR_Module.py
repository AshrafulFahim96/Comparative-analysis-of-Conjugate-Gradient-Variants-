import numpy as np


def snr_fun(sig, ref=None):
    """ x = snr(sig, ref)
        snr -- Compute Signal-to-Noise Ratio for images

        # Usage:
        x = snr(sig, ref)  -- 1st time or
        x = snr(sig)       -- afterwards

        # Input:
        sig         Modified image
        ref         Reference image

        # Output:
        x           SNR value"""

    if ref is not None:
        ref_save = ref
    else:
        ref_save = 0

    mse = np.mean(np.square(ref_save - sig))
    dv = np.mean(np.abs(ref_save - ref_save.mean()) ** 2)
    x = 10 * np.log10(dv / mse)

    return x
