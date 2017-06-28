import os
from core.discovery import extract_dependency_patterns

datafile = os.path.join('projects', '20ng', '20ng_sci', 'data', 'raw', 'train.json')
extract_dependency_patterns.extract_patterns(datafile, lower=True)