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
VERSION="base"
DATASET="RTE"

for BATCH in 96
# 128
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 14400"
        OPTS+=" --dim_ff 57600"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done

for BATCH in 200
# 384
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 8960"
        OPTS+=" --dim_ff 35840"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done


for BATCH in 112
# 176
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 12800"
        OPTS+=" --dim_ff 51200"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done

for BATCH in 180
# 320
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 9600"
        OPTS+=" --dim_ff 38400"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done

for BATCH in 260
# 448
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 7680"
        OPTS+=" --dim_ff 30720"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done

for BATCH in 768
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 5120"
        OPTS+=" --dim_ff 20480"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done

for BATCH in 960
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 4096"
        OPTS+=" --dim_ff 16384"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done

for BATCH in 1536
do
    for QUANTIZE in True False 
    # 
    do
        OPTS=""
        OPTS+=" --dataset ${DATASET}"
        OPTS+=" --base-path ${BASE_PATH}"
        OPTS+=" --model-config gpt2-${VERSION}"
        OPTS+=" --batch-size $BATCH"
        OPTS+=" --train-iters 800"
        OPTS+=" --save-iters 1000"
        OPTS+=" --max-decoder-length 128"
        OPTS+=" --save ${BASE_PATH}/results"
        OPTS+=" --save-name finetune-gpt2-ckpt"
        OPTS+=" --lr 0.00005"
        OPTS+=" --inspect-iters 100"
        OPTS+=" --warmup-iters 100"
        OPTS+=" --lr-decay-style constant"
        OPTS+=" --weight-decay 1e-3"
        OPTS+=" --clip-grad 10.0"
        OPTS+=" --loss-scale 128"
        OPTS+=" --seed 28"
        OPTS+=" --dim_model 2560"
        OPTS+=" --dim_ff 10240"
        OPTS+=" --quantize $QUANTIZE"
        # OPTS+=" --load ${BASE_PATH}/results/GPT2-${VERSION}.pt"

        CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/gpt2/finetune_gpt2.py ${OPTS}"
        echo ${CMD}

        ${CMD} 2>&1 | tee ${BASE_PATH}/logs/gpt2_superglue/finetune-gpt2-${VERSION}-${DATASET}.log
    done
done