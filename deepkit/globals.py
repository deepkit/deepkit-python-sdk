import io
from typing import Optional

import deepkit.experiment

last_experiment: Optional[deepkit.experiment.Experiment] = None

loaded_job_config = None

last_logs = io.StringIO('')
