# @Author: Xin Wen

import torch
import os
import utility
import data
import model
from option import args
from trainer_step2 import Trainer as Trainer2


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    if args.stage == 1:
        print('This is test!!!')
    else:
        t = Trainer2(args, loader, model, loss, checkpoint)
    print("Trainer prepare successed!")
    while not t.terminate():
        print("Training!!!")
        t.train()
        print("Testing!!!")
        t.test()

    checkpoint.done()

