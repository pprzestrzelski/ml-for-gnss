import matplotlib.pyplot as plt
import pandas as pd
import sys

data = pd.read_csv(sys.argv[1], sep=';')
t0 = data['Epoch'][0]
dt = data['Epoch'][1] - data['Epoch'][0]
e0 = data['Clock_bias'][0]
raw_data = data['Clock_bias'].diff().fillna(0)
scale = data['Clock_bias'].max() - data['Clock_bias'].min()
raw_data = raw_data.to_numpy() / scale

data2 = pd.read_csv(sys.argv[2], sep=';')

err = data2['Clock_bias'].values - data['Clock_bias'].values[:data2['Clock_bias'].values.shape[0]]
#plt.plot(data['Clock_bias'].values, label='Source')
#plt.plot(data2['Clock_bias'].values, label='Predicted')
plt.plot(err)
plt.ylabel('Prediction error')
plt.xlabel('Epoch')
plt.xlim(left=3000)
plt.legend()
#plt.savefig('bias.png')
plt.show()

#plt.plot(raw_data)
#plt.ylabel('Normalized bias difference')
#plt.xlabel('Readout number')
#plt.savefig('diff.png')
#plt.show()
