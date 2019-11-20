"""
For final paper plot and analysis
According to model_name and directly_month.
plot and nino3.4 .
"""
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15}
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10}
col_labels = ['Lead time', 'correlation ', 'RMSE']
table_vals = [['3 months', '0.98990', '0.11499'],
              ['6 months', '0.97161', '0.18715'],
              ['9 months', '0.93890', '0.25080'],
              ['12 months', '0.91287', '0.30619']]
my_table = plt.table(cellText=table_vals, colLabels=col_labels,
                     loc='center', cellLoc='center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('ZC Prediction Errors with Wind Stress Forced', fontsize=15, fontname='Times New Roman')
my_table.set_fontsize(20)
my_table.auto_set_column_width(3)
plt.show()
