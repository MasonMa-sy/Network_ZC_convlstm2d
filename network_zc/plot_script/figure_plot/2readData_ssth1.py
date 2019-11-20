import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from network_zc.plot_script.figure_plot import predict_369_data
from network_zc.tools import name_list, file_helper_unformatted

'''
this script for read data of zc model's output.And draw the figure of sst.
'''
# atmospheric model region
east_border = 101.25
west_border = 73.125
x_resolution = 5.625
north_border = 29
south_border = -29
y_resolition = 2

lons = np.arange(east_border + 5 * x_resolution, 360 - west_border - 2 * x_resolution + 0.5, x_resolution)
lats = np.arange(south_border + 5 * y_resolition, north_border - 5 * y_resolition + 0.5, 2)

model_name = 'conv_model_dimensionless_1_historical_sc_5@best'
if name_list.is_best:
    model_name = file_helper_unformatted.find_model_best(model_name)
model_type = name_list.model_type
is_retrain = name_list.is_retrain
is_seasonal_circle = name_list.is_seasonal_circle

prediction_month = predict_369_data.prediction_month
directly_month = predict_369_data.directly_month

start_month = 420

file_path = "D:\msy\projects\zc\zcdata\data_networks\\" + model_name + '\\' + str(directly_month) + '\\'
# meteo_file = "1toh1init.dat"

data = file_helper_unformatted.read_data_best(file_path, start_month)
sst = data[:, :, 0]
h1 = data[:, :, 1]
'''
# 获取每个变量的值
lons = fh.variables['xt_ocean'][:]
lats = fh.variables['yt_ocean'][:]
sst = fh.variables['sst_ave'][:]
#tlml = fh.variables['TLML'][:]

#tlml_units = fh.variables['TLML'].units
'''
sst_units = 'degrees C'

# 经纬度平均值
lon_0 = lons.mean()
lat_0 = lats.mean()

fig = plt.figure()

ax = fig.add_subplot(211)
ax.set_title(str(start_month))

m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100, \
            llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30, \
            resolution='l')
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)
# print("xy,%d  %d",xi,yi)

# Plot Data
# 这里我的tlml数据是24小时的，我这里只绘制第1小时的（tlml_0）
# tlml_0 = tlml[0:1:, ::, ::]
# v代表色标的范围
# v = np.linspace(-1.0, 1.0, 21, endpoint=True)
cs = m.contourf(xi, yi, np.squeeze(sst))
# cs = m.pcolor(xi, yi, np.squeeze(sst))
plt.set_cmap('jet')
# Add Grid Lines
# 绘制经纬线
m.drawparallels(np.arange(-30., 30., 10.), labels=[1, 0, 0, 0], fontsize=15)
m.drawmeridians(np.arange(100., 290., 40.), labels=[0, 0, 0, 1], fontsize=15)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(sst_units)

# Add Title
# plt.title(meteo_file)

h1_units = 'm'
ax = fig.add_subplot(212)
ax.set_title('h1')

m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100, \
            llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30, \
            resolution='l')
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)
# print("xy,%d  %d",xi,yi)

# Plot Data
# 这里我的tlml数据是24小时的，我这里只绘制第1小时的（tlml_0）
# tlml_0 = tlml[0:1:, ::, ::]
# v代表色标的范围
# v = np.linspace(-1.0, 1.0, 21, endpoint=True)
cs = m.contourf(xi, yi, np.squeeze(h1))
# cs = m.pcolor(xi, yi, np.squeeze(sst))

# Add Grid Lines
# 绘制经纬线
m.drawparallels(np.arange(-30., 30., 10.), labels=[1, 0, 0, 0], fontsize=15)
m.drawmeridians(np.arange(100., 290., 40.), labels=[0, 0, 0, 1], fontsize=15)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(h1_units)

plt.set_cmap('jet')
plt.show()
