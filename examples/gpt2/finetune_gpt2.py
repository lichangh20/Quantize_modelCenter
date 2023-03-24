import time
import random
import os
import csv

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import bmtrain as bmt

from model_center import get_args
from model_center.model import GPT2, GPT2Config
from model_center.tokenizer import GPT2Tokenizer
from model_center.dataset.gpt2dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader

def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    # model = GPT2.from_pretrained(args.model_config)
    config = GPT2Config.from_pretrained(args.model_config)
    config.num_layers = 24
    config.dim_model = args.dim_model
    config.dim_ff = args.dim_ff
    config.quantize = args.quantize
    model = GPT2(config)
    bmt.init_parameters(model)
    return model

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=args.weight_decay)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_decoder_length)
    verbalizer = torch.LongTensor(DATASET[dataset_name].get_verbalizer(tokenizer)).cuda()
    return dataset, verbalizer

def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer):
    if not args.quantize:
        # print("arrive!")
        output_dir = "{}/examples/gpt2/result/classic/{}_{}/batch={}".format(args.base_path,args.dim_model, args.dim_ff, args.batch_size)
    else:
        output_dir = "{}/examples/gpt2/result/quantize/{}_{}/batch={}".format(args.base_path,args.dim_model, args.dim_ff, args.batch_size)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, "acc.txt"), "w") as f:
        time_tuple = time.localtime(time.time())
        print('Time {}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}:'
                .format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                        time_tuple[4], time_tuple[5]), file=f)
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
        with open(os.path.join(output_dir, "acc.txt"), "a") as f:
            print(arg, getattr(args, arg), file=f)
            
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale, loss_scale_steps=100)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    print_inspect(model, '*')

    # for epoch in range(20):
    start_time = time.time()
    for epoch in range(6):
        if epoch == 1:
            epoch2_start_time = time.time()
        epoch_start_time = time.time()
        dataloader = {
            "train": DistributedDataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True),
            "dev": DistributedDataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=False),
        }

        model.train()
        for it, data in enumerate(dataloader['train']):
            input_ids = data["input_ids"]
            input_length = data["input_length"]
            labels = data["labels"]
            targets = data["targets"]
            index = data["index"]

            torch.cuda.synchronize()
            st_time = time.time()

            logits = model(input_ids, input_length, output_logits=True).logits

            loss = loss_func(logits.view(-1, logits.shape[-1]), targets.view(-1))

            logits = logits.index_select(dim=-1, index=verbalizer)
            logits = logits[torch.where(index==1)]
            loss = loss + loss_func(logits, labels)
            global_loss = bmt.sum_loss(loss).item()

            optim_manager.zero_grad()

            optim_manager.backward(loss)
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type = 2)

            optim_manager.step()

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} | time: {:.3f}".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    grad_norm,
                    elapsed_time,
                )
            )
            # if it % args.inspect_iters == 0: print_inspect(model, "*")
            # if args.save != None and it % args.save_iters == 0:
            #     bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % it)))

        model.eval()
        with torch.no_grad():
            for split in ['dev']:
                pd = []
                gt = []
                for it, data in enumerate(dataloader[split]):
                    input_ids = data["input_ids"]
                    input_length = data["input_length"]
                    labels = data["labels"]
                    index = data["index"]

                    logits = model(input_ids, input_length, output_logits=True).logits
                    logits = logits.index_select(dim=-1, index=verbalizer)
                    logits = logits[torch.where(index==1)]
                    logits = logits.argmax(dim=-1)
                
                    pd.extend(logits.cpu().tolist())
                    gt.extend(labels.cpu().tolist())

                    bmt.print_rank(
                        "{} | epoch {:3d} | Iter: {:6d}/{:6d} |".format(
                            split,
                            epoch,
                            it,
                            len(dataloader[split]),
                        )
                    )
                pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
                gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()
                bmt.print_rank(pd)
                bmt.print_rank(gt)
                
                bmt.print_rank(f"{split} epoch {epoch}:")
                if args.dataset_name in ["BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"]:
                    acc = accuracy_score(gt, pd)
                    bmt.print_rank(f"accuracy: {acc*100:.2f}")
                if args.dataset_name in ["CB"]:
                    f1 = f1_score(gt, pd, average="macro")
                    bmt.print_rank(f"Average F1: {f1*100:.2f}")
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        with open(os.path.join(output_dir, "acc.txt"),"a") as f:
            print("epoch {} time:{}".format(epoch, format(epoch_time, '.3f')), file=f)
        
    end_time = time.time()
    training_time = end_time - start_time
    training_time_without_epoch1 = end_time - epoch2_start_time
    with open(os.path.join(output_dir, "acc.txt"),"a") as f:
        print("training time", format(training_time, '.3f'), file=f)
        print("training time without epoch1", format(training_time_without_epoch1, '.3f'), file=f) 

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/superglue/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer)

if __name__ == "__main__":
    main()
