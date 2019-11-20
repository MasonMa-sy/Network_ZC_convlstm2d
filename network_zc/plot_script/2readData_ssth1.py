import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

'''
this script for read data of zc model's output.And draw the figure of sst.
Reference: https://blog.csdn.net/theonegis/article/details/50805408
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

meteo_file = "data_00000.dat"
# meteo_file = "1toh1init.dat"

sst = np.empty([20, 27])
h1 = np.empty([20, 27])
with open(meteo_file, 'rb') as fp:
    #    data = fp.read(4)
    for i in range(20):
        for j in range(27):
            data = fp.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            sst[i][j] = text
    for i in range(20):
        for j in range(27):
            data = fp.read(8)  # type(data) === bytes
            text = struct.unpack("d", data)[0]
            h1[i][j] = text
fp.close()
# while(count < 30):
# 	list_temp2 = list_temp[count].split()
# 	count = count + 1
# 	sst.append(list_temp2)
# sst = np.array(sst,dtype = 'float')

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
ax.set_title(meteo_file)

m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
            llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
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

# Add Title
# plt.title(meteo_file)

h1_units = 'm'
ax = fig.add_subplot(212)
ax.set_title('h1')

m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=100,
            llcrnrlat=-30, urcrnrlon=290, urcrnrlat=30,
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
m.drawparallels(np.arange(-30., 30., 5.), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(100., 290., 20.), labels=[0, 0, 0, 1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(h1_units)

plt.show()
