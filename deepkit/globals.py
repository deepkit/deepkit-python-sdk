import io
from typing import Optional

from deepkit.experiment import Experiment

last_experiment: Optional[Experiment] = None

loaded_job_config = None

last_logs = io.StringIO('')
