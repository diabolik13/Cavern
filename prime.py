import numpy as np

n = 91
list = []

for i in range(1,n):
    result = n/i
    if result % 1 == 0 and i % 2 != 0 and i % 3 != 0 and i % 5 != 0 and i % 7 != 0:
        list = np.append(list,i)

print(list[-1])
