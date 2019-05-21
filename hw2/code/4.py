import matplotlib.pyplot as plt
import numpy as np
import ipdb
x1 = np.linspace(0,1,100)
x2 = np.linspace(0,1,100)
x = np.meshgrid(x1,x2)
W1 = lambda x1, x2: 1-x1-x2
W2 = lambda x1, x2: x1*x2
#plt.scatter()
ipdb.set_trace()
