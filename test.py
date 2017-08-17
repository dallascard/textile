#from core.main.cross_train_and_eval_internal import cross_train_and_eval

#cross_train_and_eval('projects/mfc/immigration', 'pro_tone', 'year_group', 'config/config.json', calib_prop=0.33, nontest_prop=1.0, prefix='test', max_folds=2, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro')

#from core.main import train
#from core.experiments import dev_from_test

project_dir = 'projects/wikipedia_attacks'
print(project_dir)
subset = 'article'
model_type = 'LR'
prefix = 'p10_exclude'
label = 'label'
field_name = 'year'
config_file = 'config/default.json'

#dev_from_test.cross_train_and_eval(project_dir, subset, field_name, config_file, calib_prop=0.1, train_prop=0.1, prefix=prefix, max_folds=None, min_val=2006, max_val=2006, model_type=model_type, loss='log', do_ensemble=False, dh=0, label=label, penalty='l1', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=True, pos_label=1, average='micro', objective='f1', seed=None, use_calib_pred=False, exclude_calib=True, alpha_min=0.01, alpha_max=1000)