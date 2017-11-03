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
config_file = 'config/default.json'
do_ensemble = True
sample_labels = True

stage1_logfile = 'projects/mfc/samesex/logs/framing_Legality_year_LR_l1_100_f1.json'
annotated_subset = 'Legality_annotations'

#combo.cross_train_and_eval(project_dir, subset, 'year_group', config_file, n_calib=100, n_train=200,  model_type=model_type, loss='log', do_ensemble=do_ensemble, dh=100, label=label, penalty='l2', intercept=True, n_dev_folds=5, repeats=1, verbose=True, average='micro', objective='f1', seed=None, alpha_min=0.01, alpha_max=1000, sample_labels=sample_labels)

#fightin_words.load_and_select_features('projects/mfc/samesex/data/subsets/framing/features/unigramsfast.json', 'projects/mfc/samesex/data/subsets/Legality_annotations/features/unigramsfast.json')

#over_time_split_and_fit.test_over_time(project_dir, subset, config_file, model_type, 2012, label=label, n_calib=50, do_ensemble=True)

#preprocess_words.preprocess_words('projects/mfc/samesex', 'framing', lower=True, ngrams=2)

np.random.seed(42)
n = 200
p = 10
X = np.random.randint(low=0, high=2, size=(n, p))
X = sparse.csr_matrix(X)
beta = np.random.randn(p)

if sparse.issparse(X):
    ps = expit(X.dot(beta))
else:
    ps = expit(np.dot(X, beta))

y = np.random.binomial(p=ps, n=1)
Y = np.zeros([n, 2])
Y[:, 0] = 1 - y
Y[:, 1] = y
print(beta)

names = [str(i) for i in range(p)]
model = BLR()
model.fit(X, Y, col_names=names)
names, coefs = zip(*model.get_coefs())
print(coefs)