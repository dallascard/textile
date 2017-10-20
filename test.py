#from core.main.cross_train_and_eval_internal import cross_train_and_eval

#cross_train_and_eval('projects/mfc/immigration', 'pro_tone', 'year_group', 'config/config.json', calib_prop=0.33, nontest_prop=1.0, prefix='test', max_folds=2, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro')

#from core.main import train


#from core.experiments import combo
from core.discovery import fightin_words
from core.experiments import over_time
from core.preprocessing import preprocess_words


project_dir = 'projects/mfc/samesex'
subset = 'pro_tone'
model_type = 'LR'
label = 'label'
#field_name = 'year_group'
config_file = 'config/default.json'
do_ensemble = True
sample_labels = True

stage1_logfile = 'projects/mfc/samesex/logs/framing_Legality_year_LR_l1_100_f1.json'
annotated_subset = 'Legality_annotations'

#combo.cross_train_and_eval(project_dir, subset, 'year_group', config_file, n_calib=100, n_train=200,  model_type=model_type, loss='log', do_ensemble=do_ensemble, dh=100, label=label, penalty='l2', intercept=True, n_dev_folds=5, repeats=1, verbose=True, average='micro', objective='f1', seed=None, alpha_min=0.01, alpha_max=1000, sample_labels=sample_labels)

#fightin_words.load_and_select_features('projects/mfc/samesex/data/subsets/framing/features/unigramsfast.json', 'projects/mfc/samesex/data/subsets/Legality_annotations/features/unigramsfast.json')

#over_time.test_over_time(project_dir, subset, config_file, '1996', stage1_logfile=stage1_logfile, do_ensemble=False, label='Legality', group_identical=True, annotated_subset=annotated_subset)

preprocess_words.preprocess_words('projects/mfc/samesex', 'framing', lower=True, ngrams=2)