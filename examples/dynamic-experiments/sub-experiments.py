from time import sleep

import deepkit

experiment = deepkit.experiment(project='sub-experiments')
print('root job', experiment.id)

experiments = 10

for i in range(experiments):
    sub_experiment = experiment.create_sub_experiment()
    print('sub job', sub_experiment.id)

    sub_experiment.done()

sleep(5)
