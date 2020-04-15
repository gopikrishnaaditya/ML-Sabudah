#Problem Statement 1

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
#from sklearn import preprocessing, cross_validation
import pandas as pd

import numpy as np

np.random.seed(123)

allwalks = []

for i in range(250):
    randwalk = [0]
    for x in range(100):
        step = randwalk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2 :
            step = max(0, step - 1)

        elif dice<=5:
            step += 1

        else:
            step = step + np.random.randint(1,7)
        
    print(step)