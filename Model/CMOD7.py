import numpy as np
from scipy.interpolate import interp1d

class CMOD7():
    def __init__(self, gmf_file='gmf_cmod7_vv.dat_little_endian') -> None:
        self._read_gmf(gmf_file)

    def _read_gmf(self, gmf_file):
        self.gmf_table = np.fromfile(gmf_file, dtype=np.float32)[1:-1].reshape((250, 73, 51), order="F")

    def inverse(self, sigma0_obs, phi, incidence):
        sigma0 = np.ma.asarray(sigma0_obs)
        phi = np.ma.asarray(phi)
        incidence = np.ma.asarray(incidence)

        # 对 sigma0 中值为 0 的元素追加掩码
        sigma0.mask = sigma0.mask | (sigma0.data == 0)

        # 初始化输出数组，并合并各个输入的掩码
        wind_speed = np.ma.zeros_like(sigma0)
        wind_speed.mask = sigma0.mask | phi.mask | incidence.mask

        # 计算有效掩码（未被掩盖的点）
        valid_mask = ~wind_speed.mask
        if not np.any(valid_mask):
            return wind_speed
        
        dir_idx = np.ma.zeros(phi.shape, dtype=int)
        inc_idx = np.ma.zeros(incidence.shape, dtype=int)
        
        dir_idx[valid_mask] = np.round(phi[valid_mask] / 2.5).astype(int)
        dir_idx[valid_mask] = np.clip(dir_idx[valid_mask], 0, 72)
        
        inc_idx[valid_mask] = np.round(incidence[valid_mask] - 16).astype(int)
        inc_idx[valid_mask] = np.clip(inc_idx[valid_mask], 0, 50)
        
        wind_speeds = 0.2 * np.arange(250) + 0.2  # [0.2, 0.4, ..., 50.0]
        
        for i in range(73):
            for j in range(51):
                mask = valid_mask & (dir_idx == i) & (inc_idx == j)
                if not np.any(mask):
                    continue
                    
                gmf_curve = self.gmf_table[:, i, j]
                gmf_curve_clean, unique_indices = np.unique(gmf_curve, return_index=True)
                wind_speeds_clean = wind_speeds[unique_indices]
                target_sigma0 = sigma0[mask]
                
                # 确保 target_sigma0 是普通数组
                target_sigma0 = np.asarray(target_sigma0)
                
                # 使用线性插值计算风速
                interp_func = interp1d(
                    gmf_curve_clean, wind_speeds_clean,
                    kind='linear',
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                wind_speed[mask] = interp_func(target_sigma0)
        
        return wind_speed