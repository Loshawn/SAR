import Model
import numpy as np
from utils import func, filter
import matplotlib.pyplot as plt

# SAR 卫星方位角
platformHeading = -12.28185800368421
SAR_azimuth  = platformHeading + 90

nc_path = r"data\shenzhen_20240204T103344.nc"
phi_path = r"data\data2.nc"

# 读取数据
lon, lat, sigma0, inc = func.read_nc(
    nc_path=nc_path,
    keys=["lon", "lat", "Sigma0_VV", "IncidentAngle_VH"],
    step=50) # 原分辨率 10 m, 下采样 step = 50 分辨率 500 m 

phi = func.fit_phi(phi_path, lon_fit=lon, lat_fit=lat) - SAR_azimuth


# 反演
model = Model.CMOD5()
v = model.inverse(sigma0_obs=sigma0, phi=phi, incidence=inc, iterations=10)
v = func.remove_outliers(v)  # 去极值化, 保留前 99.5% 的风速数据
func.draw_2D(v, lon, lat, "data/CMOD5.png")

model = Model.CMOD5_N()
v = model.inverse(sigma0_obs=sigma0, phi=phi, incidence=inc, iterations=10)
v = func.remove_outliers(v)
func.draw_2D(v, lon, lat, "data/CMOD5_N.png")

model = Model.CMOD4()
v = model.inverse(sigma0_obs=sigma0, phi=phi, incidence=inc, iterations=10)
v = func.remove_outliers(v)
func.draw_2D(v, lon, lat, "data/CMOD4.png")

model = Model.CMOD_IFR2()
v = model.inverse(sigma0_obs=sigma0, phi=phi, incidence=inc, iterations=10)
v = func.remove_outliers(v)
func.draw_2D(v, lon, lat, "data/CMOD_IFR2.png")
