import os

import numpy as np

from core.discovery import extract_dependency_patterns
from core.models import blr
from core.models import ivap


#datafile = os.path.join('projects', '20ng', '20ng_sci', 'data', 'raw', 'train.json')
#extract_dependency_patterns.extract_patterns(datafile, lower=True)

#blr.main()


n = 20
x = np.arange(n)
y = np.random.rand(n)

print(x)
print(y)


ivap.isotonic_regression(x, y)
