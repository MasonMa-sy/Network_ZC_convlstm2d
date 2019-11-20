import numpy as np
import struct
import matplotlib.pyplot as plt

nino34 = []
month_start = 210
month = 100
for num in range(month_start,month_start+month):

	file_num = "%05d" % num
	meteo_file = "data_" + file_num + ".dat"

	count = 0
	sst = np.empty([20,27])
	with open(meteo_file, 'rb') as fp:
	#    data = fp.read(4)
	    for i in range(20):
	    	for j in range(27):
	    		data = fp.read(8)      #type(data) === bytes
	    		text = struct.unpack("d", data)[0]
	    		sst[i][j] = text
	fp.close()
	nino3_temp = 0
	sumi = 0
	for i in range(7, 13):
	    	for j in range(11, 20):
	    		nino3_temp+=sst[i][j]
	    		sumi=sumi+1
	nino34.append(nino3_temp/sumi)
	#print(sumi)
nino34_index = []
for num in range(1,+month-1):
	nino34_index.append((nino34[num-1]+nino34[num]+nino34[num+1])/3)
# x = np.linspace(month_start, month_start + month - 1, month)
x = np.linspace(month_start+1, month_start + month - 2, month-2)
# x = np.linspace(month_start, month_start + month - 1, month)
plt.plot(x,nino34_index)
plt.show()