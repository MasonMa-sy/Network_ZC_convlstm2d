"""
This script is for interpolate historical data.
Reference: https://blog.csdn.net/theonegis/article/details/50805408
"""
# Third-party libraries
import numpy as np
import struct
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
# My libraries
from network_zc.tools import file_helper_unformatted, name_list

data_file = name_list.data_file
data_name = name_list.data_name
data_file_statistics = name_list.data_file_statistics
data_preprocessing_file = name_list.data_preprocessing_file
data_historical_file = name_list.data_historical_file
historical_binary_file = name_list.historical_binary_file

# atmospheric model region
east_border = 101.25    # 129.375
west_border = 73.125    # 84.375
x_resolution = 5.625
north_border = 29       # 19
south_border = -29      # -19
y_resolution = 2


def plot_sst_from_netcdf(num):
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "HadISST_sst_128-277E_-20-20N.nc"
    hadisst = Dataset(filename, "r")
    # 获取每个变量的值
    lons = hadisst.variables['lon'][:]
    lats = hadisst.variables['lat'][:]
    time = hadisst.variables['time'][:]
    ssts = hadisst.variables['sst'][:]
    sst_units = hadisst.variables['sst'].units
    # 经纬度平均值
    lon_0 = lons.mean()
    lat_0 = lats.mean()

    m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
                llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
                resolution='l')
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    ssts_0 = ssts[num-1:num:, ::, ::]

    cs = m.contourf(xi, yi, np.squeeze(ssts_0))
    # cs = m.pcolor(xi, yi, np.squeeze(sst))

    # Add Grid Lines
    # 绘制经纬线
    m.drawparallels(np.arange(-30., 30., 5.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(100., 290., 20.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(sst_units)
    plt.title('Surface Air Temperature')
    plt.show()
    hadisst.close()


def plot_z20c_from_netcdf(num):
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "data_20C.nc"
    z20c = Dataset(filename, "r")
    # 获取每个变量的值
    lons = z20c.variables['X'][:]
    lats = z20c.variables['Y'][:]
    time = z20c.variables['T'][:]
    z20cs = z20c.variables['Z'][:]
    z20c_units = z20c.variables['Z'].units
    # 经纬度平均值
    lon_0 = lons.mean()
    lat_0 = lats.mean()

    m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
                llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
                resolution='l')
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    ssts_0 = z20cs[num-1:num:, ::, ::]

    cs = m.contourf(xi, yi, np.squeeze(ssts_0))
    # cs = m.pcolor(xi, yi, np.squeeze(sst))

    # Add Grid Lines
    # 绘制经纬线
    m.drawparallels(np.arange(-30., 30., 5.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(100., 290., 20.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(z20c_units)
    plt.title('Surface Air Temperature')
    plt.show()
    z20c.close()


def plot_wind_from_netcdf(num):
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "FSU_wind_stress_meridional.nc"
    wind = Dataset(filename, "r")
    # 获取每个变量的值
    lons = wind.variables['X'][:]
    lats = wind.variables['Y'][:]
    time = wind.variables['T'][:]
    winds = wind.variables['tauy'][:]
    wind_units = wind.variables['tauy'].units
    # 经纬度平均值
    lon_0 = lons.mean()
    lat_0 = lats.mean()

    m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
                llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
                resolution='l')
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    winds_0 = winds[num-1:num:, ::, ::]

    cs = m.contourf(xi, yi, np.squeeze(winds_0))
    # cs = m.pcolor(xi, yi, np.squeeze(sst))

    # Add Grid Lines
    # 绘制经纬线
    m.drawparallels(np.arange(-30., 30., 5.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(100., 290., 20.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(wind_units)
    plt.title('wind stress')
    plt.show()
    wind.close()


def plot_from_variable(sst, sst_or_z20c):
    lons_model = np.arange(east_border + 5 * x_resolution, 360 - west_border - 2 * x_resolution + 0.5, x_resolution)
    lats_model = np.arange(south_border + 5 * y_resolution, north_border - 5 * y_resolution + 0.5, y_resolution)
    lon_model, lat_model = np.meshgrid(lons_model, lats_model)
    lon_0 = lons_model.mean()
    lat_0 = lats_model.mean()
    m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
                llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
                resolution='l')
    xi, yi = m(lon_model, lat_model)

    cs = m.contourf(xi, yi, sst)
    # cs = m.pcolor(xi, yi, ssts_model)

    # Add Grid Lines
    # 绘制经纬线
    m.drawparallels(np.arange(-30., 30., 5.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(100., 290., 20.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    if sst_or_z20c == 0:
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        sst_units = 'degree C'
        cbar.set_label(sst_units)
        plt.title('Surface Sea Temperature')
    if sst_or_z20c == 1:
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        sst_units = 'm'
        cbar.set_label(sst_units)
        plt.title('Z 20 isotherm')
    plt.show()


def interpolate_sst_from_netcdf():
    # filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "HadISST_sst_128-277E_-20-20N.nc"
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "HadISST_sst_121-289E_-20-20N.nc"
    # time_start is to ensure that the num of sst is equal to z20c
    time_start = 1320
    hadisst = Dataset(filename, "r")
    # 获取每个变量的值
    lons = hadisst.variables['lon'][:]
    lats = hadisst.variables['lat'][:]
    ssts = hadisst.variables['sst'][:]

    lon, lat = np.meshgrid(lons, lats)
    # 需要插值的格点
    lons_model = np.arange(east_border + 5 * x_resolution, 360 - west_border - 2 * x_resolution + 0.5, x_resolution)
    lats_model = np.arange(south_border + 5 * y_resolution, north_border - 5 * y_resolution + 0.5, y_resolution)
    lon_model, lat_model = np.meshgrid(lons_model, lats_model)

    ssts_result = np.empty([ssts.shape[0] - time_start, 20, 27])
    for i in range(time_start, ssts.shape[0]):
        print(i)
        ssts_0 = ssts[i, :, :]
        # ssts_0 = ssts_0.flatten()
        ssts_0_mask = np.ma.getmaskarray(ssts_0)
        ssts_0 = ssts_0.compress(np.logical_not(np.ravel(ssts_0_mask)))
        lon_0 = lon.compress(np.logical_not(np.ravel(ssts_0_mask)))
        lat_0 = lat.compress(np.logical_not(np.ravel(ssts_0_mask)))

        ssts_model = griddata((lon_0, lat_0), ssts_0, (lon_model, lat_model), method='linear')
        ssts_model_nearest = griddata((lon_0, lat_0), ssts_0, (lon_model, lat_model), method='nearest')
        ssts_model_nan_index = np.isnan(ssts_model)
        ssts_model[ssts_model_nan_index] = ssts_model_nearest[ssts_model_nan_index]
        ssts_result[i-time_start, :, :] = ssts_model
        # plot_from_variable(ssts_model, 0)
    hadisst.close()
    return ssts_result
    # np.set_printoptions(threshold=np.inf)
    # print(ssts.shape[0])
    # The follow code is to plot fig of interpolated data.
    # lon_0 = lons_model.mean()
    # lat_0 = lats_model.mean()
    # m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
    #             llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
    #             resolution='l')
    # xi, yi = m(lon_model, lat_model)
    #
    # cs = m.contourf(xi, yi, ssts_model)
    # # cs = m.pcolor(xi, yi, ssts_model)
    #
    # # Add Grid Lines
    # # 绘制经纬线
    # m.drawparallels(np.arange(-30., 30., 5.), labels=[1, 0, 0, 0], fontsize=10)
    # m.drawmeridians(np.arange(100., 290., 20.), labels=[0, 0, 0, 1], fontsize=10)
    #
    # # Add Coastlines, States, and Country Boundaries
    # m.drawcoastlines()
    # m.drawstates()
    # m.drawcountries()
    #
    # # Add Colorbar
    # cbar = m.colorbar(cs, location='bottom', pad="10%")
    # sst_units = hadisst.variables['sst'].units
    # cbar.set_label(sst_units)
    # plt.title('Surface Air Temperature')
    # plt.show()


def interpolate_z20c_from_netcdf():
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "data_20C.nc"
    z20c = Dataset(filename, "r")
    # 获取每个变量的值
    lons = z20c.variables['X'][:]
    lats = z20c.variables['Y'][:]
    time = z20c.variables['T'][:]
    z20cs = z20c.variables['Z'][:]
    z20c_units = z20c.variables['Z'].units

    lon, lat = np.meshgrid(lons, lats)
    # 需要插值的格点
    lons_model = np.arange(east_border + 5 * x_resolution, 360 - west_border - 2 * x_resolution + 0.5, x_resolution)
    lats_model = np.arange(south_border + 5 * y_resolution, north_border - 5 * y_resolution + 0.5, y_resolution)
    lon_model, lat_model = np.meshgrid(lons_model, lats_model)

    z20cs_result = np.empty([z20cs.shape[0], 20, 27])
    for i in range(z20cs.shape[0]):
        print(i)
        z20cs_0 = z20cs[i, :, :]
        # ssts_0 = ssts_0.flatten()
        z20cs_0_mask = np.ma.getmaskarray(z20cs_0)
        z20cs_0 = z20cs_0.compress(np.logical_not(np.ravel(z20cs_0_mask)))
        lon_0 = lon.compress(np.logical_not(np.ravel(z20cs_0_mask)))
        lat_0 = lat.compress(np.logical_not(np.ravel(z20cs_0_mask)))

        z20cs_model = griddata((lon_0, lat_0), z20cs_0, (lon_model, lat_model), method='linear')
        z20cs_model_nearest = griddata((lon_0, lat_0), z20cs_0, (lon_model, lat_model), method='nearest')
        z20cs_model_nan_index = np.isnan(z20cs_model)
        z20cs_model[z20cs_model_nan_index] = z20cs_model_nearest[z20cs_model_nan_index]
        z20cs_result[i, :, :] = z20cs_model
        # plot_from_variable(z20cs_model, 0)

    z20c.close()
    return z20cs_result


def interpolate_wind_from_netcdf():
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "FSU_wind_stress_meridional.nc"
    wind = Dataset(filename, "r")
    # 获取每个变量的值
    lons = wind.variables['X'][:]
    lats = wind.variables['Y'][:]
    time = wind.variables['T'][:]
    winds = wind.variables['tauy'][:]
    wind_units = wind.variables['tauy'].units

    lon, lat = np.meshgrid(lons, lats)
    # 需要插值的格点
    lons_model = np.arange(east_border, 360 - west_border + 0.5, x_resolution)
    lats_model = np.arange(south_border, north_border + 0.5, y_resolution)
    lon_model, lat_model = np.meshgrid(lons_model, lats_model)

    winds_result_meridional = np.empty([winds.shape[0], 30, 34])
    for i in range(winds.shape[0]):
        print(i)
        winds_0 = winds[i, :, :].flatten()
        lon_0 = lon.flatten()
        lat_0 = lat.flatten()
        winds_model = griddata((lon_0, lat_0), winds_0, (lon_model, lat_model), method='linear')
        winds_result_meridional[i, :, :] = winds_model
    wind.close()

    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_historical_file + "FSU_wind_stress_zonal.nc"
    wind = Dataset(filename, "r")
    # 获取每个变量的值
    lons = wind.variables['X'][:]
    lats = wind.variables['Y'][:]
    time = wind.variables['T'][:]
    winds = wind.variables['taux'][:]
    wind_units = wind.variables['taux'].units

    lon, lat = np.meshgrid(lons, lats)
    winds_result_zonal = np.empty([winds.shape[0], 30, 34])
    for i in range(winds.shape[0]):
        print(i)
        winds_0 = winds[i, :, :].flatten()
        lon_0 = lon.flatten()
        lat_0 = lat.flatten()
        winds_model = griddata((lon_0, lat_0), winds_0, (lon_model, lat_model), method='linear')
        winds_result_zonal[i, :, :] = winds_model
    wind.close()
    return winds_result_meridional, winds_result_zonal


def get_mean(total):
    """
    Can both get sst mean and z20c mean.
    :param total:
    :return:
    """
    mean = np.empty([12, 20, 27])
    a_data = np.copy(total)
    for i in range(12):
        index = np.arange(i, 465, 12)
        mean[i, :, :] = np.mean(total[index], axis=0)
        # plot_from_variable(mean[i], 1)
    for i in range(a_data.shape[0]):
        a_data[i, :, :] = a_data[i, :, :] - mean[i % 12, :, :]
    return a_data


def get_mean_sstm_zc(total):
    """
    Can get sst mean by subtract sstm from zc model
    :return:
    """
    filename = "D:\msy\projects\zc\zcdata\\" + data_file + data_preprocessing_file + "sstm.dat"
    sstm = np.empty([12, 30, 34])
    with open(filename, 'rb') as fp:
        #    data = fp.read(4)
        for l in range(12):
            for i in range(30):
                for j in range(34):
                    data = fp.read(8)  # type(data) === bytes
                    text = struct.unpack("d", data)[0]
                    sstm[l][i][j] = text
    fp.close()
    mean = np.empty([12, 20, 27])
    for l in range(12):
        for i in range(5, 25):
            for j in range(5, 32):
                mean[l][i-5][j-5] = sstm[l][i][j]
    a_data = np.copy(total)
    for i in range(a_data.shape[0]):
        a_data[i, :, :] = a_data[i, :, :] - mean[i % 12, :, :]
    return a_data


def write_data(num, ssta, z20ca):
    """
    Write data named num containing data for dense.
    :param z20ca:
    :param ssta:
    :param num:
    :return:
    """
    file_num = "%05d" % num
    filename = "D:\msy\projects\zc\zcdata\\" + historical_binary_file + data_name + file_num + ".dat"
    with open(filename, 'wb') as fp:
        #    data = fp.read(4)
        for i in range(ssta.shape[0]):
            for j in range(ssta.shape[1]):
                temp = struct.pack("d", ssta[i][j])
                fp.write(temp)
        for i in range(z20ca.shape[0]):
            for j in range(z20ca.shape[1]):
                temp = struct.pack("d", z20ca[i][j])
                fp.write(temp)
    fp.close()


def out_to_file(sstas, z20cas):
    for i in range(sstas.shape[0]):
        write_data(i, sstas[i, :, :], z20cas[i, :, :])


def get_binary_data_465():
    """
    this is main function of interpolating data and save as binary data.
    :return:
    """
    sst_result = interpolate_sst_from_netcdf()
    ssta_data = get_mean(sst_result)
    z20c_result = interpolate_z20c_from_netcdf()
    z20ca_data = get_mean(z20c_result)
    print(ssta_data.shape[0], z20ca_data.shape[0])
    out_to_file(ssta_data, z20ca_data)


def get_binary_data_465_sstm():
    """
    this is main function of interpolating data and save as binary data.
    :return:
    """
    sst_result = interpolate_sst_from_netcdf()
    ssta_data = get_mean_sstm_zc(sst_result)
    z20c_result = interpolate_z20c_from_netcdf()
    z20ca_data = get_mean(z20c_result)
    print(ssta_data.shape[0], z20ca_data.shape[0])
    out_to_file(ssta_data, z20ca_data)


def get_binary_data_wind():
    wind_result_meridional, wind_result_zonal = interpolate_wind_from_netcdf()
    out_to_file(wind_result_zonal, wind_result_meridional)


if __name__ == '__main__':
    get_binary_data_wind()
    pass

