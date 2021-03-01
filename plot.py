import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "./output/plot_squared_30in_4h_1out_0d8ni.csv"
# file_path = "./output/not_optimized_30in_4h_1out_0d03ni.csv"

df = pd.read_csv(file_path, sep=',')

df.plot()

plt.show()