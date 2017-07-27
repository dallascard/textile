from core.main.cross_train_and_eval_internal import cross_train_and_eval

cross_train_and_eval('projects/mfc/immigration', 'pro_tone', 'year_group', 'config/config.json', calib_prop=0.33, nontest_prop=1.0, prefix='test', max_folds=2, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro')