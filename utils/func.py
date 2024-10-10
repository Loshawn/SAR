import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def save_csv(variable, save_path):
    """
    保存矩阵为csv文件
    """
    print("Saving... ")
    var = pd.DataFrame(variable)
    var.to_csv(save_path, index=False, header=False)
    print("Done.\n")


def read_nc(nc_path, keys: list, step=1):
    """
    读取nc文件
    输入参数:
        nc_path :   nc文件路径
        keys :      读取数据的键
        step :      数据读取步长

    输出参数:
        datas :     keys 值 的 数据列表 
    """
    filename = os.path.basename(nc_path)
    print(f"Reading data from {filename} ...")

    import netCDF4 as nc
    nf = nc.Dataset(nc_path, 'r')

    datas = []
    for key in keys:
        data = nf.variables[key]

        if len(data.shape) == 1:
            vals = data[::step]

        elif len(data.shape) == 2:
            vals = data[::step, ::step]

        elif len(data.shape) == 3:
            vals = data[0][::step, ::step]

        if hasattr(data, 'scale_factor') and hasattr(data, 'scale_factor'):
            vals = vals * data.scale_factor + data.add_offset

        datas.append(vals)
    print(f"Read Done .\n")
    return datas

def fit_phi(nc_path, lon_fit, lat_fit):
    """
    拟合phi
    输入参数:
        nc_path :   nc文件路径
        lon_fit :   需要拟合的经度数组
        lat_fit :   需要拟合的纬度数组

    输出参数:
        phi_fit :   拟合的 phi 方位角
    """
    print(f"Fitting phi ...")
    from scipy.interpolate import Rbf

    lon, lat, u10, v10 = read_nc(nc_path,
                                 ['longitude', 'latitude', 'u10', 'v10'])
    phi = np.pi / 2 - np.arctan(v10 / u10)
    # 拟合与插值
    lon, lat = np.meshgrid(lon, lat)
    rbf = Rbf(lon, lat, phi, function='multiquadric')

    if len(lon_fit.shape) == 1 and len(lat_fit.shape) == 1:
        lon_fit, lat_fit = np.meshgrid(lon_fit, lat_fit)
        phi_fit = rbf(lon_fit, lat_fit)
        print(f"Fit Done .\n")
        return phi_fit
    else:
        raise Exception("The lon_fit and lat_fit must be one-dimensional data")
    
def read_tif(tif_path: str, size=2000):
    """
    从tif文件中读取块, 返回numpy矩阵。

    参数：
        tif_path: 要读取的tif文件地址。
        size: 读取区块的大小，输入类型为：
                int 从中心取 size * size 大小的块。\n
                [heigth, width] 从中心取 heigth * width 大小的块。\n
                [x, y, heigth, width] 以 (x, y) 为起点取 heigth * width 大小的块。\n
    """
    print("Reading...")

    import rasterio
    with rasterio.open(tif_path) as src:
        total_width = src.width
        total_height = src.height
        if isinstance(size, int):
            if size == -1:
                x_start = y_start = 0
                width = total_width
                height = total_height
            else:
                x_start = (total_width - size) // 2
                y_start = (total_height - size) // 2
                width = height = size
        elif isinstance(size, (list, tuple)):
            width = size[-1]
            height = size[-2]
            if len(size) == 2:
                x_start = (total_width - width) // 2
                y_start = (total_height - height) // 2
            if len(size) == 4:
                x_start = size[0]
                y_start = size[1]

        window = rasterio.windows.Window(x_start, y_start, width, height)

        data = src.read(window=window)
        count = src.count
    print("Done.\n")
    print(f"{count} bands in total. \n"
          f"Total size: {total_height} * {total_width} \n"
          f"Selected size: {height} * {width}\n")
    return data


def read_csv(csv_path):
    """
    读取csv文件
    """
    data = pd.read_csv(csv_path)
    return data.to_numpy()


def draw(data, longitude=None, latitude=None, save_path=None, is_show=False):
    """
    绘制2D分布图。
    输入相同大小的三个二维矩阵, 经纬度可不填。
    """
    print("Drawing... ")
    if len(longitude.shape) == 1 and len(latitude.shape) == 1:
        X, Y = np.meshgrid(longitude, latitude)

    if len(longitude.shape) == 2 and len(latitude.shape) == 2:
        latitude = np.ma.masked_equal(latitude, 0)
        longitude = np.ma.masked_equal(longitude, 0)

        x = np.linspace(np.min(longitude), np.max(longitude),
                        longitude.shape[1])
        y = np.linspace(np.max(latitude), np.min(latitude), latitude.shape[0])
        X, Y = np.meshgrid(x, y)

    if longitude is None and latitude is None:
        plt.imshow(data, cmap='jet')
        plt.axis('off')
    else:
        plt.pcolormesh(X, Y, data, vmin=0, cmap='jet')
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        
    plt.axis('equal')
    plt.colorbar(label='Speed (m/s)')
    plt.title('Wind speed distribution')
    print("Done.\n")

    if save_path is not None:
        print("Saving picture...")
        plt.savefig(save_path, dpi=1000)
        print("Done.\n")

    if is_show:
        plt.show()
    plt.close()



def remove_outliers(data, threshold=0.995):
    """
    去极值化, 保留前99.5%的数值
    输入参数:
        data :      原始数据
        threshold : 阈值 

    输出参数:
        data :   去极值化后的数据
    """
    hist, bin_edges = np.histogram(data.compressed(), bins=np.arange(0,41,0.001),density=True)
    cumulative_prob = np.cumsum(hist) * 0.001
    # 找到 99% 的阈值
    threshold_index = np.argmax(cumulative_prob >= threshold)
    threshold_value = bin_edges[threshold_index]
    data[data>threshold_value] = threshold_value
    return data