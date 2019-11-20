import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
'''
this script for read data of zc model's output.And draw the figure of sst.
'''
#atmospheric model region
east_border = 101.25
west_border = 73.125
x_resolution = 5.625
north_border = 29
south_border = -29
y_resolition = 2

lons = np.arange(east_border,360 - west_border + 0.5,x_resolution)
lats = np.arange(south_border,north_border + 0.5,2)

num = 0

file_num = "%05d" % num
model_file = "data_in_" + file_num + ".dat"
file_num = "%05d" % (num)
model_file2 = "data_out_" + file_num + ".dat"

# v代表色标的范围
#v = np.linspace(-0.1, 0.1, 21, endpoint=True)

#first fig
fh = open(model_file, mode='r')
list_temp = []
for line in fh:
	list_temp.append(line)
fh.close()

count = 0
sst = []
while(count < 30):
	list_temp2 = list_temp[count].split()
	count = count + 1
	sst.append(list_temp2)
sst = np.array(sst,dtype = 'float')

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
ax.set_title(model_file)

m = Basemap(lat_0=lat_0, lon_0=lon_0,llcrnrlon = 100,   \
	llcrnrlat = -30, urcrnrlon = 290, urcrnrlat = 30,	\
	resolution = 'l')
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)
#print("xy,%d  %d",xi,yi)

# Plot Data
# 这里我的tlml数据是24小时的，我这里只绘制第1小时的（tlml_0）
# tlml_0 = tlml[0:1:, ::, ::]
cs = m.contourf(xi, yi, np.squeeze(sst))
bs = m.contourf(xi, yi, np.squeeze(sst))
#cs = m.contourf(xi, yi, np.squeeze(sst), v)
#bs = m.contourf(xi, yi, np.squeeze(sst), v)
#cs = m.pcolor(xi, yi, np.squeeze(sst))s

# Add Grid Lines
# 绘制经纬线
m.drawparallels(np.arange(-30., 30., 5.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(100., 290., 20.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
#cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=v)
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(sst_units)

# sencond fig
fh = open(model_file2, mode='r')
list_temp = []
for line in fh:
	list_temp.append(line)
fh.close()

count = 0
sst = []
while(count < 30):
	list_temp2 = list_temp[count].split()
	count = count + 1
	sst.append(list_temp2)
sst = np.array(sst,dtype = 'float')

ax = fig.add_subplot(212)
ax.set_title(model_file2)

m = Basemap(lat_0=lat_0, lon_0=lon_0,llcrnrlon = 100,   \
	llcrnrlat = -30, urcrnrlon = 290, urcrnrlat = 30,	\
	resolution = 'l')
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)
#print("xy,%d  %d",xi,yi)

# Plot Data
# 这里我的tlml数据是24小时的，我这里只绘制第1小时的（tlml_0）
# tlml_0 = tlml[0:1:, ::, ::]

cs = m.contourf(xi, yi, np.squeeze(sst))
bs = m.contourf(xi, yi, np.squeeze(sst))
#cs = m.contourf(xi, yi, np.squeeze(sst), v)
#bs = m.contourf(xi, yi, np.squeeze(sst), v)
#cs = m.pcolor(xi, yi, np.squeeze(sst))s

# Add Grid Lines
# 绘制经纬线
m.drawparallels(np.arange(-30., 30., 5.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(100., 290., 20.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
#cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=v)
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(sst_units)

fig.tight_layout()
# Add Title
plt.show()
