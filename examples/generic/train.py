import random

import deepkit

context = deepkit.context()
context.add_file(__file__)

test = deepkit.create_metric('test')

for i in range(100):
    deepkit.set_info(i, random.random())

total = 1_000_000;
for i in range(1_000_000):
    test.send(i, random.random())
    deepkit.epoch(i, total)

print("Bye.")