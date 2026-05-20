import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class CommonAxisInterpolator:
    @staticmethod
    def interpolate_to_common_axis(dataframes, num_points=512):
        """
        Interpolates all dataframes to a common logarithmic frequency axis.
        """
        if not dataframes:
            return {}

        # Find common frequency range
        min_freqs = [df['frequency'].min() for df in dataframes]
        max_freqs = [df['frequency'].max() for df in dataframes]
        
        global_min = max(20.0, max(min_freqs))
        global_max = min(20000.0, min(max_freqs))
        
        # Create common log-spaced frequency axis
        common_freqs = np.logspace(np.log10(global_min), np.log10(global_max), num_points)
        
        interpolated_mags = []
        raw_mags = []

        for df in dataframes:
            f = interp1d(df['frequency'], df['magnitude'], kind='linear', fill_value="extrapolate")
            mag_interp = f(common_freqs)
            interpolated_mags.append(mag_interp)
            raw_mags.append(mag_interp) # In this context, raw refers to the unnormalized but interpolated data

        mags_2d = np.array(interpolated_mags)
        
        # Normalize magnitudes for algorithms that perform better in 0-1 range
        mag_min = mags_2d.min()
        mag_max = mags_2d.max()
        norm_mags_2d = (mags_2d - mag_min) / (mag_max - mag_min) if mag_max > mag_min else mags_2d
        
        return {
            'common_freqs': common_freqs,
            'norm_freqs': common_freqs, # Often useful to have a direct reference
            'norm_mags_2d': norm_mags_2d,
            'raw_mags_2d': mags_2d,
            'mag_min': mag_min,
            'mag_max': mag_max
        }

    @staticmethod
    def unnormalize_magnitude(norm_mag, mag_min, mag_max):
        """Converts normalized 0-1 magnitude back to original dB scale."""
        return norm_mag * (mag_max - mag_min) + mag_min
