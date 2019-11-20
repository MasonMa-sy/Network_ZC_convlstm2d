import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

east_border = 101.25
west_border = 73.125
x_resolution = 5.625
north_border = 29
south_border = -29
y_resolition = 2

lons = np.arange(east_border + 5 * x_resolution, 360 - west_border - 2 * x_resolution + 0.5, x_resolution)
lats = np.arange(south_border + 5 * y_resolition, north_border - 5 * y_resolition + 0.5, 2)

width_sst = 360 - west_border - 2 * x_resolution - east_border - 5 * x_resolution
height_sst = north_border - 5 * y_resolition - south_border - 5 * y_resolition

lon_0 = lons.mean()
lat_0 = lats.mean()

fig = plt.figure()
ax = fig.add_subplot(111)

m = Basemap(lat_0=lat_0, lon_0=lon_0, llcrnrlon=east_border,
            llcrnrlat=south_border, urcrnrlon=360 - west_border, urcrnrlat=north_border,
            resolution='l')
m.drawparallels(np.arange(-30., 30., 10.), labels=[1, 0, 0, 0], fontsize=15)
m.drawmeridians(np.arange(100., 290., 40.), labels=[0, 0, 0, 1], fontsize=15)

# x = np.arange(360-150, 360-90+0.5, 60)
# y = np.arange(-5, 5+0.5, 10)
# x, y = np.meshgrid(x, y)
# xi, yi = m(x, y)
# f = np.abs(x) + np.abs(y) - 1
# cs = m.contourf(xi, yi, f)
rect_nino = plt.Rectangle((190, -5), 50, 10, fill=False, color='r', linewidth=5)
rect_sst = plt.Rectangle((east_border + 5 * x_resolution, south_border + 5 * y_resolition), width_sst, height_sst,
                         fill=False, color='b', linewidth=3)
ax.add_patch(rect_nino)
ax.add_patch(rect_sst)

m.drawcoastlines()

plt.show()

