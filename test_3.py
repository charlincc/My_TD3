from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import random
import tensorflow_probability as tfp
import math

state_low = np.array([2., 6.0, 9.5, 0.2, 0.2, 0.2, 0.2])
state_high = np.array([5., 9.0, 12.0, 0.8, 0.8, 0.8, 0.8])
