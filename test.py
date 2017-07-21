import os

import numpy as np

from core.discovery import extract_dependency_patterns
from core.models import blr
from core.models import ivap
from core.preprocessing import preprocess_characters2


#datafile = os.path.join('projects', '20ng', '20ng_sci', 'data', 'raw', 'train.json')
#extract_dependency_patterns.extract_patterns(datafile, lower=True)

#blr.main()

preprocess_characters2.preprocess_chracters('projects/mfc/immigration', 'pro_tone', False, 3, 1000)
