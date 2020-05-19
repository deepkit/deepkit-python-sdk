import random
import deepkit

experiment = deepkit.experiment()
experiment.add_file(__file__)

test = experiment.define_metric('test')

for i in range(10):
    experiment.set_info(i, random.random())

total = 100_000

for i in range(total):
    test.send(i, random.gauss(25, 25/3))
    experiment.epoch(i, total)

print("Bye.")
