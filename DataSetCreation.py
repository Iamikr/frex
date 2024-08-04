import os
import pandas as pd
from Scripts.utils import *

processed = []
directory1 = os.listdir('DataReady')
for folder in directory1:
    directory2 = os.listdir('DataReady/{}'.format(folder))
    for filename in directory2:
         if filename.endswith('csv'):
                processed.append(filename)
print(processed)
directory1 = os.listdir('Data')
for folder in directory1:
    directory2 = os.listdir('Data/{}'.format(folder))
    for filename in directory2:
        if filename.endswith('csv') and filename not in processed:
            print('Data/{}/{}'.format(folder,filename))
            columns = ['Date','Open','High','Low','Close','Volume']
            index = ['Date']
            data = generateData('Data/{}/{}'.format(folder,filename), columns, index, indicators = True)
            data.to_csv('DataReady/{}/{}'.format(folder,filename))