import os

import numpy as np

from core.discovery import extract_dependency_patterns
from core.models import blr
from core.models import ivap
from core.preprocessing import preprocess_characters2

from core.main.cross_train_and_eval_internal import cross_train_and_eval


#datafile = os.path.join('projects', '20ng', '20ng_sci', 'data', 'raw', 'train.json')
#extract_dependency_patterns.extract_patterns(datafile, lower=True)

#blr.main()

#preprocess_characters2.preprocess_chracters('projects/mfc/immigration', 'pro_tone', False, 3, 1000)

#python -m core.main.cross_train_and_eval_internal pro_tone year_group config/config.json --max_folds 2
cross_train_and_eval('projects/mfc/immigration', 'pro_tone', 'year_group', 'config/config.json', calib_prop=0.33, nontest_prop=1.0, prefix='test', max_folds=2, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro')