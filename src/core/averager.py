import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize
from tslearn.metrics import SquaredEuclidean, SoftDTW
from tslearn.barycenters import dtw_barycenter_averaging
import concurrent.futures

def _sdtw_worker(Z, X_i, gamma):
    D = SquaredEuclidean(Z, X_i)
    sdtw = SoftDTW(D, gamma=gamma)
    return sdtw.compute(), D.jacobian_product(sdtw.grad())

def _softdtw_func(Z, X, weights, barycenter_shape, gamma, progress_queue=None, state=None, max_iter=25):
    Z_reshaped = Z.reshape(barycenter_shape)
    G = np.zeros_like(Z_reshaped)
    obj = 0.0
    
    total_curves = len(X)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_sdtw_worker, Z_reshaped, X[i], gamma): i for i in range(total_curves)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            val, G_tmp = future.result()
            G += weights[i] * G_tmp
            obj += weights[i] * val
            completed += 1
            if progress_queue is not None and state is not None:
                progress_queue.put(("sub_progress", (state['iter'], completed, total_curves, max_iter)))
            
    return obj, G.ravel()

class CurveAverager:
    @staticmethod
    def arithmetic_mean(norm_mags_2d: np.ndarray) -> np.ndarray:
        """Standard point-by-point mean on normalized data."""
        return np.mean(norm_mags_2d, axis=0)

    @staticmethod
    def geometric_arc_length_blend(norm_freqs: np.ndarray, norm_mags_2d: np.ndarray, y_weight: float = 0.15) -> np.ndarray:
        """
        Geometric blend treating responses as paths and averaging normalized arc lengths.
        y_weight controls stiffness (0.0 = Arithmetic Mean, 1.0 = Full Arc Length).
        """
        if len(norm_mags_2d) == 0:
            return np.array([])
        if len(norm_mags_2d) == 1:
            return norm_mags_2d[0]
            
        num_points = len(norm_freqs)
        
        # CRITICAL FIX: Normalize the X-axis for the geometric space (0 to 1)
        # Without this, the X distance (e.g. 200Hz step) completely dwarfs the Y distance (e.g. 0.01 magnitude step)
        log_f = np.log10(norm_freqs)
        x_geom = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
        
        # A shared uniform parametric timeline
        t_standard = np.linspace(0, 1, num_points)
            
        def blend_two(mag_A, mag_B, weight_A, weight_B):
            def calc_arc_length(x, y):
                if len(x) != len(y):
                    min_len = min(len(x), len(y))
                    x, y = x[:min_len], y[:min_len]

                # APPLY Y-WEIGHT HERE in the fully normalized [0,1] x [0,1] space
                dist = np.hypot(np.diff(x), np.diff(y) * y_weight)
                arc = np.concatenate(([0], np.cumsum(dist)))
                
                if arc[-1] > 0:
                    return arc / arc[-1]
                return arc

            # Calculate arc length using the normalized geometric X-axis
            arc_A = calc_arc_length(x_geom, mag_A)
            arc_B = calc_arc_length(x_geom, mag_B)

            def interp_standard(arc, val):
                u_arc, uniq = np.unique(arc, return_index=True)
                if len(u_arc) < 2:
                    return np.full(num_points, val[0])
                # Direct evaluation at the standard number of points
                return PchipInterpolator(u_arc, val[uniq], extrapolate=True)(t_standard)

            # Map both curves to the shared t_standard timeline
            interp_x_A = interp_standard(arc_A, x_geom)
            interp_y_A = interp_standard(arc_A, mag_A)
            
            interp_x_B = interp_standard(arc_B, x_geom)
            interp_y_B = interp_standard(arc_B, mag_B)

            # Geometric Blend
            blend_x_norm = (interp_x_A * weight_A) + (interp_x_B * weight_B)
            blend_y_norm = (interp_y_A * weight_A) + (interp_y_B * weight_B)

            # Re-interpolate the blended parametric curve back onto the rigid frequency grid
            u_blend_x, uniq_blend = np.unique(blend_x_norm, return_index=True)
            if len(u_blend_x) < 2:
                return np.full(num_points, blend_y_norm[0] if len(blend_y_norm) > 0 else 0)

            # Interpolate against x_geom to find the Y values at the original x_geom grid points
            final_mags = PchipInterpolator(u_blend_x, blend_y_norm[uniq_blend], extrapolate=True)(x_geom)
            return final_mags

        current_avg = norm_mags_2d[0]
        for i in range(1, len(norm_mags_2d)):
            weight_B = 1.0 / (i + 1)
            weight_A = 1.0 - weight_B
            current_avg = blend_two(current_avg, norm_mags_2d[i], weight_A, weight_B)
            
        return current_avg

    @staticmethod
    def soft_dtw_barycenter(norm_mags_2d: np.ndarray, gamma: float = 1.0, progress_queue = None) -> np.ndarray:
        """Smooth shape-preserving mean using Regularized Soft-DTW."""
        if len(norm_mags_2d) == 0:
            return np.array([])
            
        ts_data = np.expand_dims(norm_mags_2d, axis=-1)
        weights = np.ones(len(ts_data)) / len(ts_data)
        
        # Initialize with Euclidean barycenter (arithmetic mean)
        init_barycenter = np.mean(ts_data, axis=0)
        
        max_iter = 25
        state = {'iter': 0}
        total_curves = len(ts_data)
        
        if progress_queue is not None:
            progress_queue.put(("init_progress", max_iter * total_curves))
        
        def callback(xk):
            state['iter'] += 1

        res = minimize(
            _softdtw_func, 
            init_barycenter.ravel(), 
            args=(ts_data, weights, init_barycenter.shape, gamma, progress_queue, state, max_iter),
            method="L-BFGS-B", 
            jac=True, 
            tol=1e-3,
            callback=callback,
            options=dict(maxiter=max_iter)
        )
        
        barycenter = res.x.reshape(init_barycenter.shape)
        return barycenter.flatten()

    @staticmethod
    def wasserstein_barycenter(norm_freqs: np.ndarray, norm_mags_2d: np.ndarray) -> np.ndarray:
        """Stable shape-preserving average using Optimal Transport (1D Wasserstein)."""
        if len(norm_mags_2d) == 0:
            return np.array([])
        
        num_points = len(norm_freqs)
        common_quantiles = np.linspace(0, 1, num_points)
        
        inv_cdfs = []
        
        for mag in norm_mags_2d:
            # 1. Compute PDF and CDF
            mag_positive = np.clip(mag, a_min=1e-10, a_max=None)
            cdf = np.cumsum(mag_positive)
            cdf_normalized = cdf / cdf[-1]
            
            # Prepend (0, freq[0]-epsilon) to properly anchor the CDF to 0 at the start
            cdf_padded = np.insert(cdf_normalized, 0, 0.0)
            freqs_padded = np.insert(norm_freqs, 0, norm_freqs[0] - 1e-6)
            
            # 2. Compute Inverse CDF (Quantile function)
            # Remove duplicates in CDF for strict monotonicity in interpolation
            _, unique_indices = np.unique(cdf_padded, return_index=True)
            inv_cdf = interp1d(cdf_padded[unique_indices], freqs_padded[unique_indices], 
                               bounds_error=False, fill_value=(freqs_padded[0], freqs_padded[-1]))(common_quantiles)
            inv_cdfs.append(inv_cdf)
            
        # 3. Average the aligned inverse CDFs
        mean_inv_cdf = np.mean(inv_cdfs, axis=0)
        
        # Ensure strict monotonicity for inversion
        mean_inv_cdf = np.maximum.accumulate(mean_inv_cdf)
        mean_inv_cdf += np.linspace(0, 1e-9, num_points)
        
        # 4. Map back to original frequency grid to get the mean CDF
        # We interpolate quantiles as a function of the frequencies
        mean_cdf = interp1d(mean_inv_cdf, common_quantiles, bounds_error=False, fill_value=(0, 1))(norm_freqs)
        
        # 5. Density is the derivative of the CDF. Using np.diff accurately recovers the bin mass.
        barycenter_mag = np.diff(np.insert(mean_cdf, 0, 0.0))
        
        # Clean up any potential NaNs or infs from edge cases
        barycenter_mag = np.nan_to_num(barycenter_mag, nan=0.0, posinf=0.0, neginf=0.0)
        barycenter_mag = np.clip(barycenter_mag, a_min=0, a_max=None)
        
        # Re-normalize to [0, 1]
        mag_min = np.min(barycenter_mag)
        mag_max = np.max(barycenter_mag)
        if mag_max > mag_min:
            barycenter_mag = (barycenter_mag - mag_min) / (mag_max - mag_min)
        else:
            barycenter_mag = np.zeros_like(barycenter_mag)
        
        return barycenter_mag

    @staticmethod
    def hard_dtw_barycenter(norm_freqs: np.ndarray, norm_mags_2d: np.ndarray, radius_percent: float = 2.5) -> np.ndarray:
        """
        Shape-preserving mean using Hard-DTW (DBA) with Sakoe-Chiba constraints.
        Radius is calculated dynamically as a percentage of the total frequency points.
        """
        if len(norm_mags_2d) == 0:
            return np.array([])
            
        # Calculate absolute points from percentage
        dynamic_radius = max(1, int(len(norm_freqs) * (radius_percent / 100.0)))
            
        # tslearn expects 3D shape: (n_ts, sz, d)
        ts_data = np.expand_dims(norm_mags_2d, axis=-1)
        
        # Calculate the barycenter with Sakoe-Chiba constraints
        barycenter = dtw_barycenter_averaging(
            X=ts_data,
            max_iter=15,
            metric_params={
                "global_constraint": 2, 
                "sakoe_chiba_radius": dynamic_radius
            }
        )
        
        return barycenter.flatten()
