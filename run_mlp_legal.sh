OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/climate/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r0
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/climate/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r1
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/climate/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r2
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/guncontrol/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r0
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/guncontrol/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r1
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/guncontrol/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r2
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/samesex/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r0
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/samesex/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r1
OMP_NUM_THREADS=4 python -m core.experiments.combo projects/mfc/samesex/ framing year_group config/doc2vec_test.json --label Legality --model MLP --n_train 800 --n_calib 100 --penalty l2 --suffix _r2

