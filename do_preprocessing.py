import os
from core.preprocessing import preprocess_labels
from core.preprocessing import preprocess_words

subset = 'pro_tone'
base_project = os.path.join('projects', 'mfc')
subprojects = ['climate', 'guncontrol', 'immigration', 'samesex', 'smoking']
for s in subprojects:
    project = os.path.join(base_project, s)
    print(project)
    preprocess_labels.preprocess_labels(project, subset, label_name='label', metadata_fields=['year_group'])
    preprocess_words.preprocess_words(project, subset, lower=True)
