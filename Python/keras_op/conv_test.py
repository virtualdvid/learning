from itertools import chain, product
 
conv1 = [32]
conv2 = [32, 64, 128, 256]
 
products = (product(conv1, *([conv2] * i)) for i in range(0, len(conv2)))

for combo in chain.from_iterable(products):
	print(combo)
