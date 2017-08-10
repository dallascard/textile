#from core.main.cross_train_and_eval_internal import cross_train_and_eval

#cross_train_and_eval('projects/mfc/immigration', 'pro_tone', 'year_group', 'config/config.json', calib_prop=0.33, nontest_prop=1.0, prefix='test', max_folds=2, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro')

from core.main import train

project_dir = 'projects/mfc/immigration/'
subset = 'pro_tone_train'
model_type = 'MLP'
model_name = 'test'
label = 'label'
feature_defs = ['unigrams,min_df=1,transform=doc2vec']

train.train_model(project_dir, model_type, model_name, subset, label, feature_defs)
