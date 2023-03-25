# QuantizeModelCenter

The code is based [ModelCenter](https://github.com/OpenBMB/ModelCenter) and [Cuda_LSS_HQ](https://github.com/lichangh20/Cuda_LSS_HQ)

## INSTALL

Tested with PyTorch 1.12.1 + CUDA 11.3. On a 8-card A100 machine.

Step 1: Install Cuda_LSS_HQ

​	Follow [Readme](https://github.com/lichangh20/Cuda_LSS_HQ#readme)

​	

## BERT

result is at ./examples/bert/result/*

```
bash ./examples/bert/RTE.sh 
```



## GPT2

result is at ./examples/gpt2/result/*

```
bash ./examples/gpt2/RTE.sh 
```