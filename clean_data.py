import pandas as pd
import numpy as np

data_input = np.array(pd.read_csv("data/OLHS_input_5324"))
data_output = np.array(pd.read_csv("data/OLHS_output_5324"))

clean_input = np.array([])
clean_output = np.array([])

for i in data_output:

    if i[1] < 1:

        clean_output = np.append(clean_output,i)
        clean_input = np.append(clean_input,data_input[ int( i[0]) ] )
    if i[0] < 0:
        print('sb')



clean_input = np.reshape(clean_input,(-1,data_input.shape[1]))
clean_output = np.reshape(clean_output,(-1,data_output.shape[1]))
