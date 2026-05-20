import numpy as np

class OutlierDetector:
    @staticmethod
    def filter_outliers(norm_mags_2d, threshold_std=1.0):
        """
        Identifies curves that deviate from the mean by more than the threshold.
        """
        if len(norm_mags_2d) <= 2:
            return list(range(len(norm_mags_2d))), []

        mean_mag = np.mean(norm_mags_2d, axis=0)
        
        # Calculate the 'distance' of each curve from the mean
        # We'll use the root-mean-square of the difference
        diffs = norm_mags_2d - mean_mag
        rms_diffs = np.sqrt(np.mean(diffs**2, axis=1))
        
        # Calculate the standard deviation of these distances
        std_dist = np.std(rms_diffs)
        mean_dist = np.mean(rms_diffs)
        
        kept_indices = []
        excluded_indices = []
        
        for i, dist in enumerate(rms_diffs):
            # If a curve is further from the mean than (threshold * group_std)
            if dist > mean_dist + (threshold_std * std_dist):
                excluded_indices.append(i)
            else:
                kept_indices.append(i)
                
        return kept_indices, excluded_indices
