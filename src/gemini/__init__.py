#!/usr/bin/env python

import os
gemini_home = os.path.dirname(os.path.abspath(__file__))
__home__ = os.environ.get('GEMINI_HOME', str(gemini_home))
__datasets__ = os.environ.get('GEMINI_DATASETS', f'{__home__}/datasets/')

#==============================================================================

from .base import Base as Gemini
from .opt  import GeminiOpt
