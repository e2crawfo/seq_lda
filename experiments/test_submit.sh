python ihmm_synthetic.py --quick --parallel --name=ihmm_synthetic --use-time=0 --directory='built_experiments'
mv ihmm_synthetic.zip built_experiments
sd-submit --task=cv --n-jobs=60 --input-zip=built_experiments/ihmm_synthetic.zip --n-nodes=1 --ppn=4 --walltime=0:15:00 --add-date=1 --verbose=1 --show-script=1 --exclude="train_scratch" --test=1 --scratch=/data/seq_lda
sd-submit --task=test --n-jobs=30 --input-zip=built_experiments/ihmm_synthetic_cv.zip --n-nodes=1 --ppn=4 --walltime=0:15:00 --add-date=1 --verbose=1 --show-script=1 --exclude="test_scratch" --test=1 --scratch=/data/seq_lda