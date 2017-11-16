import numpy as np
from scipy import sparse
from scipy.special import expit

#from core.experiments import combo
from core.discovery import fightin_words
from core.experiments import over_time, over_time_split_and_fit
from core.preprocessing import preprocess_words
from core.models.blr import BLR


project_dir = 'projects/mfc/samesex'
subset = 'framing'
model_type = 'LR'
label = 'Legality'
#field_name = 'year_group'
config_file = 'config/config.json'
do_ensemble = True
sample_labels = True

#stage1_logfile = 'projects/mfc/samesex/logs/framing_Legality_year_LR_l1_100_f1.json'
#annotated_subset = 'Legality_annotations'

#combo.cross_train_and_eval(project_dir, subset, 'year_group', config_file, n_calib=100, n_train=200,  model_type=model_type, loss='log', do_ensemble=do_ensemble, dh=100, label=label, penalty='l2', intercept=True, n_dev_folds=5, repeats=1, verbose=True, average='micro', objective='f1', seed=None, alpha_min=0.01, alpha_max=1000, sample_labels=sample_labels)

#fightin_words.load_and_select_features('projects/mfc/samesex/data/subsets/framing/features/unigramsfast.json', 'projects/mfc/samesex/data/subsets/Legality_annotations/features/unigramsfast.json')

over_time_split_and_fit.test_over_time(project_dir, subset, config_file, model_type, 'year', 2011, 2012, label=label, n_train=1000, n_calib=0, do_ensemble=True, sample_labels=False)

#preprocess_words.preprocess_words('projects/mfc/samesex', 'framing', lower=True, ngrams=2)
