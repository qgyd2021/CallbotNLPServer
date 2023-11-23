import os
import time

from project_settings import project_path
from toolbox.design_patterns.singleton import ParamsSingleton


class Table(ParamsSingleton):
    _initialized = False
