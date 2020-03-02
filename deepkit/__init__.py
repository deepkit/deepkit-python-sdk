import sys

import deepkit.globals
import deepkit.utils
from deepkit.client import Client
from deepkit.experiment import Experiment, ExperimentOptions


def log(s):
    if deepkit.globals.last_experiment:
        deepkit.globals.last_experiment.log(s)
    else:
        deepkit.globals.last_logs.write(s)


def experiment(project=None, account=None) -> Experiment:
    """
    :param project: If the current folder is not linked and you don't specify a project here, an error is raised since
                    Deepkit isn't able to know to which project the experiments data should be sent.
    :param account: Per default the account linked to this folder is used (see `deepkit link`),
                    this is on a new system `localhost`.
                    You can overwrite which account is used by specifying the name here (see `deepkit id` for
                    available accounts in your system).
    :return:
    """
    """
    :param options: ExperimentOptions
    :return: returns either a new experiemtn or the last created one. Never creates multiple experiments.
    """
    if deepkit.globals.last_experiment:
        return deepkit.globals.last_experiment

    return Experiment(ExperimentOptions(project=project, account=account))


if deepkit.utils.in_self_execution():
    class StdHook:
        def __init__(self, s):
            self.s = s

        def fileno(self):
            return self.s.fileno()

        def isatty(self):
            return self.s.isatty()

        def flush(self):
            self.s.flush()

        def write(self, s):
            self.s.write(s)
            log(s)


    sys.stdout = StdHook(sys.__stdout__)
    sys.stderr = StdHook(sys.__stderr__)
