import os
from core.preprocessing import preprocess_labels

subset = 'pro_tone'
base_project = os.path.join('projects', 'mfc')
subprojects = ['climate', 'guncontrol', 'immigration', 'samesex', 'smoking']
for s in subprojects:
    project = os.path.join(base_project, s)
    preprocess_labels.preprocess_labels(project, subset, label_name='label', metadata_fields=['year_group'])