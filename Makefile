# Default setup: not loading existing checkpoints
# To Load existing checkpoints, specify LOAD_MODEL=1 and CHECKPOINT="pathtocheck"

export LOAD_MODEL = 0
export CHECKPOINT = ""
export BATCH_SIZE = 64
export MODEL_NAME = model_0
export NUM_EPOCHS = 1000

prepare_data:
	sbatch scripts/prepare_data.sbatch

train:
	python py/train.py

train_cluster:
	sbatch scripts/train.sbatch

test:
	python py/test.py

test_cluster:
	sbatch scripts/test.sbatch
