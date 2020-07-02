# this script starts multiple experiments
import random
import threading
from time import sleep

import deepkit

experiment_optimization_id = '1'

hyper_parameters_base = {
    'lr': 0.1,
    'optimizer': 'adam',
}

root_experiment = deepkit.experiment(project='threaded')
experiments = 10


class ExperimentExecutor(threading.Thread):
    def __init__(self, id: int, root_experiment: deepkit.Experiment, hyper_parameters: dict):
        super().__init__()
        self.daemon = True
        self.id = id
        self.root_experiment = root_experiment
        self.hyper_parameters = hyper_parameters

    def run(self):
        experiment = self.root_experiment.create_sub_experiment()
        experiment.set_info('id', id)
        experiment.set_info('optimization_id', experiment_optimization_id)
        experiment.set_full_config(hyper_parameters)
        experiment.add_file(__file__)

        total = 1_000
        for epoch in range(total):
            experiment.log_metric('test', random.gauss(25, 25 / 3), x=epoch)
            experiment.epoch(epoch + 1, total)
            sleep(0.05)

        if self.id == 2:
            experiment.set_description('Aborted on purpose')
            experiment.abort()
        else:
            experiment.done()

        print(f"Experiment #{self.id} ended.")


threads = []
for i in range(experiments):
    hyper_parameters = hyper_parameters_base.copy()
    hyper_parameters['lr'] += i * 0.1  # poor man's hyper-parameter optimization :o)

    executor = ExperimentExecutor(i, root_experiment, hyper_parameters)
    executor.start()
    threads.append(executor)

for executor in threads:
    executor.join()

print("All done")
