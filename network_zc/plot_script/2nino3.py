import numpy as np
import struct
import matplotlib.pyplot as plt

nino3 = []
for num in range(1200):

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
	    	for j in range(15, 26):
	    		nino3_temp+=sst[i][j]
	    		sumi=sumi+1
	nino3.append(nino3_temp/sumi)
plt.plot(nino3)
plt.show()