import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

'''
this script for read data of zc model's output.And draw the figure of sst.
Reference: https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
'''
# atmospheric model region
east_border = 101.25
west_border = 73.125
x_resolution = 5.625
north_border = 29
south_border = -29
y_resolition = 2

lons = np.arange(east_border, 360 - west_border + 0.5, x_resolution)
lats = np.arange(south_border, north_border + 0.5, 2)

meteo_file = "mean\sstm.dat"

sstm = np.empty([30, 34, 12])

with open(meteo_file, 'rb') as fp:
    #    data = fp.read(4)
    for l in range(12):
        for i in range(30):
            for j in range(34):
                data = fp.read(8)  # type(data) === bytes
                text = struct.unpack("d", data)[0]
                sstm[i][j][l] = text
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

fig = plt.figure(figsize=(10, 13))
# fig = plt.figure(figsize=(50,50))
# fig.set_figheight(100)
# fig.set_figwidth(100)
v = np.linspace(16, 30, 8, endpoint=True)
for l in range(12):
    ax = fig.add_subplot(6, 2, l + 1)
    ax.set_title("sstm for %d month" % (l + 1))
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
    cs = m.contourf(xi, yi, np.squeeze(sstm[:, :, l]), v)
    # cs = m.pcolor(xi, yi, np.squeeze(sst))

    # Add Grid Lines
    # 绘制经纬线
    m.drawparallels(np.arange(-30., 30., 10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(100., 290., 40.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    # cbar = m.colorbar(cs, location='bottom', pad="10%")
    # cbar.set_label(sst_units)

plt.subplots_adjust(left=None, bottom=None, right=0.8, top=None,
                    wspace=None, hspace=None)
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(cs, cax=cb_ax, ticks=v)
cbar.set_label(sst_units)

# plt.savefig("1test.png")
plt.show()
