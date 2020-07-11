
# Run training
# $ bash bin/train.sh <model>

# Kwargs
MODEL=${1:-rssm}

# Settings
CUDA=0
BATCH_SIZE=16
MAX_STEPS=1000
TEST_INTERVAL=100
SAVE_INTERVAL=100
SEED=0

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${MODEL}

# Dataset path
export DATASET_DIR=./data/
export DATASET_NAME=mnist

# Config for training
export CONFIG_PATH=./examples/config.json

python3 examples/train.py --cuda ${CUDA} --model ${MODEL} \
    --seed ${SEED} --batch-size ${BATCH_SIZE} --max-steps ${MAX_STEPS} \
    --test-interval ${TEST_INTERVAL} --save-interval ${SAVE_INTERVAL}
