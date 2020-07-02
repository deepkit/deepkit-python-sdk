import os
import sys

import deepkit.globals
import deepkit.utils
from deepkit.client import Client
from deepkit.experiment import Experiment, ExperimentOptions
import getpass

from deepkit.home import get_home_config


def log(s):
    if deepkit.globals.last_experiment:
        deepkit.globals.last_experiment.log(s)
    else:
        deepkit.globals.last_logs.write(s)


def experiment(project=None, account=None) -> Experiment:
    """
    Per default this method returns a singleton.

    If you start an experiment using the Deepkit cli (`deepkit run`) or the Deepkit app, the experiment
    is created beforehand and this method picks it up. If an experiment is run without cli or app,
    then this method creates a new one. In any case, this method returns always the same instance, so
    you don't strictly need to save or pass around its return value.

    If you want to create new sub experiments you should use:

        import deepkit
        root_experiment = deepkit.experiment()
        sub_experiment = root_experiment.create_sub_experiment()

    This will create _always_ a new child experiments. In this cases, make sure to call `experiment.done()`,
    (or abort, crashed, failed) manually to end the created experiment and pass around the created experiment
    instance manually (since its not tracked).

    :param project: If the current folder is not linked and you don't specify a project here, an error is raised since
                    Deepkit isn't able to know to which project the experiments data should be sent.
    :param account: Per default the first account linked to this folder is used (see `deepkit link` or `deepkit-sdk auth -l`),
                    this is on a new system `localhost`.
                    You can overwrite which account is used by specifying the name here (see `deepkit id` for
                    available accounts in your system).
    :return:
    """
    """
    :return: returns either a new experiment or the last created one.
    """
    if not deepkit.globals.last_experiment:
        deepkit.globals.last_experiment = Experiment(project=project, account=account, monitoring=True,
                                                     try_pick_up=True)

    return deepkit.globals.last_experiment


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


def get_credentials():
    username = input("Username: ")
    try:
        password = getpass.getpass()
        return username, password
    except Exception as error:
        print('ERROR', error)


access_key_map = dict()


def access_key_map_cache_key(host, port, ssl):
    return host + '-' + str(port) + str(ssl)


def login(
        access_key=None,
        host='app.deepkit.ai',
        port=443,
        ssl=True,
):
    """
    In environments (like Jupyter Notebooks/Google Colab) where its not possible to use the Deepkit CLI to authenticate
    with a Deepkit server (deepkit auth) or where "deepkit run" is not used, it's required to provide an access-key
    or login via username/password.

    It's important to call this method BEFORE deepkit.experiment() is called.
    """
    if host is 'localhost':
        ssl = False

        if port == 443:
            port = 8960

        try:
            config = get_home_config()
            account_config = config.get_account_for_name('localhost')
            access_key = account_config.token
        except Exception:
            pass

    if access_key is None:
        cache_key = access_key_map_cache_key(host, port, ssl)
        if cache_key in access_key_map:
            access_key = access_key_map[cache_key]
        else:
            print("No access_key provided. Please provide username and password.")
            print(
                f"Note: You can create an access_key directly in the CLI using `deepkit access-key {host} --port {port}`")
            client = Client()
            client.host = host
            client.port = port
            client.ssl = ssl

            username, password = get_credentials()

            print(f"Connecting {client.host}:{client.port}")
            client.connect_anon()
            access_key = client.app_action_threadsafe('login', [username, password]).result()
            if not access_key:
                raise Exception("Credentials check failed")

            print("Login successful. Access key is " + access_key)
            access_key_map[cache_key] = access_key

    os.environ['DEEPKIT_HOST'] = host
    os.environ['DEEPKIT_SSL'] = '1' if ssl else '0'
    os.environ['DEEPKIT_PORT'] = str(port)

    if 'DEEPKIT_JOB_ACCESSTOKEN' in os.environ:
        del os.environ['DEEPKIT_JOB_ACCESSTOKEN']

    if 'DEEPKIT_JOB_ID' in os.environ:
        del os.environ['DEEPKIT_JOB_ID']

    os.environ['DEEPKIT_ACCESSTOKEN'] = access_key
