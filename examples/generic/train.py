import random
import deepkit
from time import sleep

experiment = deepkit.experiment()
experiment.add_file(__file__)

test = experiment.define_metric('test')

for i in range(10):
    experiment.set_info(str(i), random.random())

total = 1_000

for i in range(total):
    test.send(random.gauss(25, 25/3), x=i)
    experiment.epoch(i, total)
    sleep(0.005)

print("Bye.")
