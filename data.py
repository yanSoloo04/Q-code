from ucimlrepo import fetch_ucirepo 
import numpy as np
import panda as pd
  
# fetch dataset 
htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
X = htru2.data.features 
y = htru2.data.targets 
  
# metadata 
print(X) 
  
# variable information 
print(y)