import time
import numpy as np

iterations = 10000
print('iterations:', iterations)

# start = time.time()
# for _ in range(iterations):
#     np.random.randint(2)
# a = (start - time.time())/iterations
# print(a)

# start = time.time()
# for _ in range(iterations):
#     np.random.choice([True, False])
# b = (start - time.time())/iterations
# print(b)

# start = time.time()
# for _ in range(iterations):
#     np.random.choice([1, 0])
# c = (start - time.time())/iterations
# print(c)

# print(a < b)
# print(a < c)

# print(b < a)
# print(b < c)

# print(c < a)
# print(c < b)

# from random import uniform

# list_dicts = [{'test': -uniform(0, 1)} for _ in range(iterations)]

# start = time.time()
# for _ in range(iterations):
#     min_val = min([d['test'] for d in list_dicts])
# a = (start - time.time())/iterations
# print('min:', min_val)
# print(a)

# start = time.time()
# min_val = 0
# for _ in range(iterations):
#     for d in list_dicts:
#         if d['test'] < min_val:
#             min_val = d['test']
# b = (start - time.time())/iterations
# print('min:', min_val)
# print(b)

# print(a < b)

hello = 'hello'
world = 'world'

print(f'{hello} \
     {world}')