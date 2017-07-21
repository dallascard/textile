import os

import numpy as np

from core.discovery import extract_dependency_patterns
from core.models import blr
from core.models import ivap
from core.preprocessing import preprocess_labels


#datafile = os.path.join('projects', '20ng', '20ng_sci', 'data', 'raw', 'train.json')
#extract_dependency_patterns.extract_patterns(datafile, lower=True)

#blr.main()

preprocess_labels.preprocess_labels('projects/mfc/immigration', 'pro_tone', 'label')
