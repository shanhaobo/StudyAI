from functools import partial

import random

from tqdm import tqdm, notebook
from time import sleep

def add(x, y):
    return x + y

a1 = 1
b1 = 10

a2 = 20
b2 = 30

def randab():
    return random.randint(a2, b2)

rand_add = partial(add, random.randint(a1, b1), randab())
rand_add_2 = partial(add, a2, b2)

print(rand_add())
print(rand_add_2())

a2 = 600
b2 = 900

print(rand_add())
print(rand_add_2())

for i in tqdm.notebook.trange(1000, desc="Loop"):
    sleep(0.01)
