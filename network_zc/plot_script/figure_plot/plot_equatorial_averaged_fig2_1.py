import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from network_zc.tools import file_helper_unformatted


def get_sst_equatorial(data_2d):
    sst_equatorial = []
    for j in range(27):
        sst_equatorial_temp = 0
        for i in range(7, 13):
            sst_equatorial_temp += data_2d[i][j][0]
        sst_equatorial.append(sst_equatorial_temp/6)
    return sst_equatorial


def get_sst_equatorial_from_data(file_num1, month1):
    sst_equatorial = np.empty([month1, grid_length])
    for i in range(file_num1, file_num1+month1):
        data = file_helper_unformatted.read_data_sstaha(i)
        sst_equatorial[i-file_num1] = get_sst_equatorial(data)
    return sst_equatorial


# free 1200
# forced 192
# historical 0
file_num = 1200
month = 240

grid_length = 27

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

sst_equatorials = get_sst_equatorial_from_data(file_num, month)

print(sst_equatorials)
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(0, grid_length-1, grid_length, dtype=int)
y = np.linspace(0, month-1, month, dtype=int)
plt.xlim(0, grid_length-1)
plt.ylim(0, month-1)
x_ticks1 = np.linspace(1.8888, 23.2222, 4)
x_ticks2 = ['140E', '180', '140W', '100W']
plt.xticks(x_ticks1, x_ticks2)

str_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# revise1
# y_ticks1 = np.arange(0, month, 12, dtype=int)
# y_ticks2 = np.arange(file_num, file_num+month, 12, dtype=int)
y_ticks1 = np.arange(0, month, 12, dtype=int)
# y_ticks2 = ['Jan-'+str(x) for x in np.arange(80, 100, 1, dtype=int)]
y_ticks2 = ['Jan-'+str(x) for x in np.arange(100, 120, 1, dtype=int)]
plt.yticks(y_ticks1, y_ticks2)

v = np.linspace(-3.0, 6.0, 10, endpoint=True)
plt.contourf(x, y, sst_equatorials)
plt.set_cmap('jet')
# plt.contour(x, y, ssta_error_all, 20)
position = fig.add_axes([0.15, 0.05, 0.7, 0.03])
plt.colorbar(cax=position, orientation='horizontal')

ax.set_title('Free ZC data', fontsize=12, fontname='Times New Roman')
# ax.set_title('Forced ZC data', fontsize=12, fontname='Times New Roman')
# ax.set_title('Historical data', fontsize=12, fontname='Times New Roman')

plt.show()
