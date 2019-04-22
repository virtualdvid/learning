import time
import numpy as np

iterations = 1000000

start = time.time()
for _ in range(iterations):
    a = np.random.randint(2)
print((start - time.time())/iterations)

start = time.time()
for _ in range(iterations):
    b = np.random.choice([True, False])
print((start - time.time())/iterations)

start = time.time()
for _ in range(iterations):
    c = np.random.choice([1, 0])
print((start - time.time())/iterations)


print(a < b)
print(a < c)

print(b < a)
print(b < c)

print(c < a)
print(c < b)
