#from core.main.cross_train_and_eval_internal import cross_train_and_eval

#cross_train_and_eval('projects/mfc/immigration', 'pro_tone', 'year_group', 'config/config.json', calib_prop=0.33, nontest_prop=1.0, prefix='test', max_folds=2, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro')

#from core.main import train


from core.experiments import no_split

project_dir = 'projects/mfc/samesex'
subset = 'pro_tone'
model_type = 'LR'
label = 'label'
#field_name = 'year_group'
config_file = 'config/default.json'
do_ensemble = True

no_split.cross_train_and_eval(project_dir, subset, config_file, n_train=200,  model_type=model_type, loss='log', do_ensemble=do_ensemble, dh=100, label=label, penalty='l2', intercept=True, n_dev_folds=5, repeats=1, verbose=True, average='micro', objective='f1', seed=None, alpha_min=0.01, alpha_max=1000)


