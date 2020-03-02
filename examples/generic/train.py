import random
import deepkit

experiment = deepkit.experiment()
experiment.add_file(__file__)

test = experiment.define_metric('test')

for i in range(100):
    experiment.set_info(i, random.random())

total = 10_000

for i in range(total):
    test.send(random.gauss(25, 25/3), x=i)
    experiment.epoch(i, total)

print("Bye.")
