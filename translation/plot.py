import numpy as np
import matplotlib.pyplot as plt

all_losses=np.load('result/losses_1.npy')
plt.figure()
plt.plot(all_losses)
plt.show()
