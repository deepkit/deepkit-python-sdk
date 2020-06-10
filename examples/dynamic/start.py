# this script starts multiple experiments
import random
import threading

import deepkit

experiment_optimization_id = '1'

hyper_parameters_base = {
    'lr': 0.1,
    'optimizer': 'adam',
}

experiments = 10


class ExperimentExecutor(threading.Thread):
    def __init__(self, id: int, experiment: deepkit.Experiment, hyper_parameters: dict):
        super().__init__()
        self.id = id
        self.experiment = experiment
        self.hyper_parameters = hyper_parameters

        experiment.set_info('id', id)
        experiment.set_list('dynamic')
        experiment.set_info('optimization_id', experiment_optimization_id)
        experiment.set_full_config(hyper_parameters)
        experiment.add_file(__file__)

    def run(self):
        total = 1_000
        for epoch in range(total):
            self.experiment.log_metric('test', random.gauss(25, 25 / 3), x=epoch)
            self.experiment.epoch(epoch + 1, total)

        if self.id == 2:
            self.experiment.set_description('Aborted on purpose')
            self.experiment.abort()
        else:
            self.experiment.done()

        print(f"Experiment #{self.id} ended.")


for i in range(experiments):
    hyper_parameters = hyper_parameters_base.copy()
    hyper_parameters['lr'] += i * 0.1  # :o)
    experiment = deepkit.experiment(new=True)
    executor = ExperimentExecutor(i, experiment, hyper_parameters)
    executor.start()
