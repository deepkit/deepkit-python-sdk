import io
from typing import Optional

import deepkit.context

last_context: Optional[deepkit.context.Context] = None

loaded_job = None

last_logs = io.StringIO('')
