import time
import os

import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

import bmtrain as bmt
import json

from model_center import get_args
from model_center.model import Bert, BertConfig
from model_center.tokenizer import BertTokenizer
from model_center.dataset.bertdataset import DATASET
from model_center.utils import print_inspect
from model_center.layer import Linear
from model_center.dataset import DistributedDataLoader
from model_center.layer import qconfig

class BertModel(torch.nn.Module):
    def __init__(self, args, num_types):
        super().__init__()
        config = BertConfig.from_pretrained("bert-large-uncased")
        #Todo
        config.num_layers = 24
        config.dim_model = args.dim_model
        config.dim_ff = args.dim_ff
        config.quantize = args.quantize
        self.bert : Bert = Bert(config)
        bmt.init_parameters(self.bert)
        # self.bert : Bert = Bert.from_pretrained(args.model_config)
        dim_model = self.bert.input_embedding.dim_model
        self.dense = Linear(dim_model, num_types)
        bmt.init_parameters(self.dense)

    def forward(self, *args, **kwargs):
        pooler_output = self.bert(*args, **kwargs, output_pooler_output=True).pooler_output
        assert pooler_output.dtype == torch.float16
        logits = self.dense(pooler_output)
        # test_logit = torch.rand_like(logits) * 1e-3
        # test_logit.requires_grad_(True)
        return logits

def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    num_types = {
        "BoolQ" : 2,
        "CB" : 3,
        "COPA" : 1,
        "RTE" : 2,
        "WiC" : 2,
    }
    model = BertModel(args, num_types[args.dataset_name])
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
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_encoder_length)
    return dataset


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    if not args.quantize:
        # print("arrive!")
        output_dir = "{}/examples/bert/result/classic/{}_{}/batch={}".format(args.base_path,args.dim_model, args.dim_ff, args.batch_size)
    else:
        output_dir = "{}/examples/bert/result/quantize/{}_{}/batch={}".format(args.base_path,args.dim_model, args.dim_ff, args.batch_size)
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

    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    print_inspect(model, '*')

    # for epoch in range(12):
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
            if args.dataset_name == 'COPA':
                input_ids0 = data["input_ids0"]
                attention_mask0 = data["attention_mask0"]
                token_type_ids0 = data["token_type_ids0"]
                input_ids1 = data["input_ids1"]
                attention_mask1 = data["attention_mask1"]
                token_type_ids1 = data["token_type_ids1"]
                labels = data["labels"]
            else:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                token_type_ids = data["token_type_ids"]
                labels = data["labels"]

            torch.cuda.synchronize()
            st_time = time.time()

            if args.dataset_name == 'COPA':
                logits = torch.cat([
                    model(input_ids0, attention_mask=attention_mask0, token_type_ids=token_type_ids0),
                    model(input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1),
                ], dim=1)
            else:
                logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

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

        model.eval()
        with torch.no_grad():
            for split in ['dev']:
                pd = []
                gt = []
                for it, data in enumerate(dataloader[split]):
                    if args.dataset_name == 'COPA':
                        input_ids0 = data["input_ids0"]
                        attention_mask0 = data["attention_mask0"]
                        token_type_ids0 = data["token_type_ids0"]
                        input_ids1 = data["input_ids1"]
                        attention_mask1 = data["attention_mask1"]
                        token_type_ids1 = data["token_type_ids1"]
                        labels = data["labels"]
                        logits = torch.cat([
                            model(input_ids0, attention_mask=attention_mask0, token_type_ids=token_type_ids0),
                            model(input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1),
                        ], dim=1)
                    else:
                        input_ids = data["input_ids"]
                        attention_mask = data["attention_mask"]
                        token_type_ids = data["token_type_ids"]
                        labels = data["labels"]
                        logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                    loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
                    logits = logits.argmax(dim=-1)
                    pd.extend(logits.cpu().tolist())
                    gt.extend(labels.cpu().tolist())

                    bmt.print_rank(
                        "{} | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f}".format(
                            split,
                            epoch,
                            it,
                            len(dataloader[split]),
                            loss,
                        )
                    )

                pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
                gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()
                
                bmt.print_rank(f"{split} epoch {epoch}:")
                if args.dataset_name in ["BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"]:
                    acc = accuracy_score(gt, pd)
                    bmt.print_rank(f"accuracy: {acc*100:.2f}")
                if args.dataset_name in ["CB"]:
                    rcl = f1_score(gt, pd, average="macro")
                    f1 = recall_score(gt, pd, average="macro")
                    bmt.print_rank(f"recall: {rcl*100:.2f}")
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
        
    # if args.quantize:
    #     with open(os.path.join(output_dir, "time.json"), "w") as f:
    #         Dict = {}
    #         Dict["forward"] = qconfig.forward
    #         Dict["grad_weight"] = qconfig.grad_weight
    #         Dict["grad_input"] = qconfig.grad_input
    #         Dict["hadamard"] = qconfig.hadamard
    #         Dict["special"] = qconfig.special_layer
    #         Dict["scale"] = qconfig.scale
    #         full_time_list = []
    #         iterate_key = ["forward", "grad_weight", "grad_input"]
    #         for keys in iterate_key:
    #             fullTime = 0
    #             for timesKey in Dict[keys].keys():
    #                 if timesKey in ["method1", "method2", "method3"]:
    #                     continue
    #                 else:
    #                     fullTime += Dict[keys][timesKey]
    #             full_time_list.append(fullTime)
    #         Dict["forward_other"] = qconfig.linear_forward - qconfig.hadamard - full_time_list[0] - qconfig.scale
    #         Dict["forward_total"] = qconfig.linear_forward
    #         Dict["backward_other"] = qconfig.linear_backward - full_time_list[1] - full_time_list[2]
    #         Dict["backward_total"] = qconfig.linear_backward
    #         Dict["linear_total"] = qconfig.linear_forward + qconfig.linear_backward
    #         Dict["otherLayer_total"] = training_time - Dict["linear_total"]
    #         json.dump(Dict, f, indent=4)


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/superglue/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
