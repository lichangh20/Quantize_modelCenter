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

BASE_PATH="/root/autodl-tmp/lichangh20/modelCenter/ModelCenter"
VERSION="bert-large-cased"
DATASET="RTE"

for BATCH in 160
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --model-config ${VERSION}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --dataset_name ${DATASET}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --warmup-iters 40"
        OPTS+=" --lr 0.00005"
        OPTS+=" --max-encoder-length 128"
        OPTS+=" --train-iters 400"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-2"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 14400"
        OPTS+=" --dim_ff 57600"
        OPTS+=" --quantize $QUANTIZE"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/bert_superglue/finetune-${VERSION}-${DATASET}.log
    done
done


for BATCH in 480
# 
do
    for QUANTIZE in True False
    # 
    do
        OPTS=""
        OPTS+=" --model-config ${VERSION}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --dataset_name ${DATASET}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --warmup-iters 40"
        OPTS+=" --lr 0.00005"
        OPTS+=" --max-encoder-length 128"
        OPTS+=" --train-iters 400"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-2"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 8960"
        OPTS+=" --dim_ff 35840"
        OPTS+=" --quantize $QUANTIZE"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/bert_superglue/finetune-${VERSION}-${DATASET}.log
    done
done


for BATCH in 224
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --model-config ${VERSION}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --dataset_name ${DATASET}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --warmup-iters 40"
        OPTS+=" --lr 0.00005"
        OPTS+=" --max-encoder-length 128"
        OPTS+=" --train-iters 400"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-2"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 12800"
        OPTS+=" --dim_ff 51200"
        OPTS+=" --quantize $QUANTIZE"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/bert_superglue/finetune-${VERSION}-${DATASET}.log
    done
done

for BATCH in 416
# 
do
    for QUANTIZE in True False
    # 
    do
        OPTS=""
        OPTS+=" --model-config ${VERSION}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --dataset_name ${DATASET}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --warmup-iters 40"
        OPTS+=" --lr 0.00005"
        OPTS+=" --max-encoder-length 128"
        OPTS+=" --train-iters 400"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-2"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 9600"
        OPTS+=" --dim_ff 38400"
        OPTS+=" --quantize $QUANTIZE"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/bert_superglue/finetune-${VERSION}-${DATASET}.log
    done
done

for BATCH in 600
# 
do
    for QUANTIZE in True False
    # 
    do
        OPTS=""
        OPTS+=" --model-config ${VERSION}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --dataset_name ${DATASET}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --warmup-iters 40"
        OPTS+=" --lr 0.00005"
        OPTS+=" --max-encoder-length 128"
        OPTS+=" --train-iters 400"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-2"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 7680"
        OPTS+=" --dim_ff 30720"
        OPTS+=" --quantize $QUANTIZE"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/bert_superglue/finetune-${VERSION}-${DATASET}.log
    done
done

for BATCH in 960
# 
do
    for QUANTIZE in True False
    # 
    do
        OPTS=""
        OPTS+=" --model-config ${VERSION}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --dataset_name ${DATASET}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --warmup-iters 40"
        OPTS+=" --lr 0.00005"
        OPTS+=" --max-encoder-length 128"
        OPTS+=" --train-iters 400"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-2"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 5120"
        OPTS+=" --dim_ff 20480"
        OPTS+=" --quantize $QUANTIZE"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/bert_superglue/finetune-${VERSION}-${DATASET}.log
    done
done