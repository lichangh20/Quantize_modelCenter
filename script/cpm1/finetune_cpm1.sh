#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/home/hx/ModelCenter"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${BASE_PATH}/configs/cpm1/cpm1-large"
OPTS+=" --batch-size 64"
OPTS+=" --train-iters 3000"
OPTS+=" --save-iters 1000"
OPTS+=" --max-length 256"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-cpm1-ckpt"
OPTS+=" --lr 0.02"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 200"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-3"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
# OPTS+=" --load ${BASE_PATH}/results/CPM1-new.pt"
OPTS+=" --load ${BASE_PATH}/results/noam-1e-3-0.1-checkpoint-9000.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/finetune_cpm1.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/finetune-cpm1-new-test.log