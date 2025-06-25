import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import attention_BCE_loss, attention_CDC_loss
from torchvision import transforms
import cv2
import torch.nn.functional as F
from datasets.dataset_MicroUS import MicroUS_dataset, MultiscaleGenerator
from utils import *
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import albumentations as A
import optuna



def trainer_MicroUS(args, model, snapshot_path):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    hard_weight = args.weight

    db_train = MicroUS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [MultiscaleGenerator(split = "train", output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn, drop_last=True)
    valloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn, drop_last=True)


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # print(model)

    #Segmentattion training parameters
    #Probati AdamW
    optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch_seg = args.max_epochs_seg
    max_iterations = args.max_epochs_seg * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    seg_iterator = tqdm(range(max_epoch_seg), ncols=70)

    #Classification training parameters


    criterion_cls=nn.BCEWithLogitsLoss()
    max_epoch_cls = args.max_epochs_cls
    cls_iterator=tqdm(range(max_epoch_cls), ncols=70)

    train_segmentation(args,model, trainloader, valloader, optimizer, scheduler, seg_iterator, hard_weight,writer,max_epoch_seg, snapshot_path,iter_num)

    # train_classification(args, model, trainloader, optimizer, scheduler, cls_iterator, base_lr ,criterion_cls, writer, max_epoch_cls, max_iterations, snapshot_path, iter_num)
    # with open("logit_logs.txt","w") as f:

    #     for epoch_num in iterator:
    #         epoch_loss = []
    #         for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
    #             image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['non_expert'], sampled_batch['non_expert0'], sampled_batch['non_expert1'], sampled_batch['non_expert2']
    #             image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), non_expert_batch.cuda(), non_expert0_batch.cuda(), non_expert1_batch.cuda(), non_expert2_batch.cuda()
    #             # Calculate cls_label_batch
    #             cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
    #             outputs, out0, out1, out2, cls_output = model(image_batch)
    #             outputs = torch.sigmoid(outputs).squeeze(dim=1)
    #             out0 = torch.sigmoid(out0).squeeze(dim=1)
    #             out1 = torch.sigmoid(out1).squeeze(dim=1)
    #             out2 = torch.sigmoid(out2).squeeze(dim=1)

    #             loss0 = attention_BCE_loss(hard_weight, label0_batch, out0, non_expert0_batch, ks=1)
    #             loss1 = attention_BCE_loss(hard_weight, label1_batch, out1, non_expert1_batch, ks=2)
    #             loss2 = attention_BCE_loss(hard_weight, label2_batch, out2, non_expert2_batch, ks=3)
    #             loss3 = attention_BCE_loss(hard_weight, label_batch, outputs, non_expert_batch, ks=4)
    #             #print(f"cls_loss:{cls_loss}")
    #             cls_output = cls_output.squeeze(-1)
    #             f.write(f"Cls_Label_batch_val:  {cls_label_batch}")
    #             f.write(f"Cls_output: {torch.sigmoid(cls_output)}")
    #             cls_loss = criterion_cls(cls_output, cls_label_batch.float())
           
    #             #For  classification
    #             loss = loss0 + loss1 + loss2 + loss3 + 0.065*cls_loss
    #             #Standard
    #             #loss = loss0 + loss1 + loss2 + loss3

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
                
    #             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = lr_

    #             iter_num = iter_num + 1
    #             epoch_loss.append(loss)
    #             writer.add_scalar('info/lr', lr_, iter_num)
    #             writer.add_scalar('info/total_loss', loss, iter_num)

    #         average_loss = sum(epoch_loss)/len(epoch_loss)
    #         logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))

    #         if epoch_num >= max_epoch - 1:
    #             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
    #             torch.save(model.state_dict(), save_mode_path)
    #             logging.info("save model to {}".format(save_mode_path))
    #             iterator.close()
    #             break

    
    
            
            

    writer.close()
    logging.shutdown()

    return "Training Finished!"



def nested_cv_with_optuna(args, model_fn, dataset, snapshot_path, n_trials=1, outer_folds=3, inner_folds=2):

    def evaluate_dice(model, val_loader):
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].cuda()
                label = batch['label'].cuda()

                output, *_ = model(image)
                pred = (torch.sigmoid(output.squeeze(1)) > 0.5).float()
    
                for i in range(len(pred)):
                    if pred[i].sum(dim=(0,1)) == 0:  # No foreground in prediction
                        if label[i].sum(dim=(0,1)) == 0:
                            dices.append(1.0)
                        else:
                            dices.append(0.0)
                    else:
                        if label[i].sum(dim=(0,1)) == 0:
                            dices.append(0.0)
                        else:
                            dices.append(metric.binary.dc(pred[i].cpu().numpy(), label[i].cpu().numpy()))
        model.train()
        return np.mean(dices)
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    outer_kf = KFold(n_splits=outer_folds, shuffle=False)
    outer_scores = []

    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_kf.split(dataset)):
        logging.info(f"\n== Outer Fold {outer_fold+1}/{outer_folds} ==")

        outer_train_val = Subset(dataset, train_val_idx)
        outer_test = Subset(dataset, test_idx)

        def objective(trial):
            base_lr = trial.suggest_float("base_lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [2,4])
            pool_method = trial.suggest_categorical("pool_method", ["avgpool", "maxpool"])
            weight = trial.suggest_int("weight", 2, 10)
            heads_csa = trial.suggest_int("heads_csa", 2,15)

            args.base_lr = base_lr
            args.weight_decay = weight_decay
            args.batch_size = batch_size
            args.pool_method = pool_method
            args.weight = weight   
            args.heads_csa = heads_csa      

            inner_kf = KFold(n_splits=inner_folds, shuffle=False)
            inner_scores = []
            inner_fold_counter = 1

            for inner_train_idx, inner_val_idx in inner_kf.split(outer_train_val):
                logging.info(f"\n== Inner Fold {inner_fold_counter}/{inner_folds} ==")
                inner_train = Subset(outer_train_val, inner_train_idx)
                inner_val = Subset(outer_train_val, inner_val_idx)

                trainloader = DataLoader(inner_train, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
                valloader = DataLoader(inner_val, batch_size=4, shuffle=False, num_workers=2)

                model_instance = model_fn(args)

                if args.n_gpu > 1:
                    model_instance = nn.DataParallel(model_instance)

                optimizer = optim.SGD(model_instance.parameters(), lr=base_lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
                writer = SummaryWriter(snapshot_path + f'/log_trial_{trial.number}_fold_{outer_fold}')

                iter_num = 0
                seg_iterator = tqdm(range(args.max_epochs_seg), ncols=70)
                cls_iterator = tqdm(range(args.max_epochs_cls), ncols=70)
                criterion_cls = nn.BCEWithLogitsLoss()

                train_segmentation(args, model_instance, trainloader, valloader, optimizer, scheduler, seg_iterator, args.weight,
                                   writer, args.max_epochs_seg, snapshot_path, iter_num)
                # train_classification(args, model, trainloader, optimizer, scheduler, cls_iterator, base_lr, criterion_cls,
                #                      writer, args.max_epochs_cls, args.max_epochs_cls * len(trainloader), snapshot_path, iter_num)


                score = evaluate_dice(model_instance, valloader)
                inner_scores.append(score)
                inner_fold_counter += 1

            return sum(inner_scores) / len(inner_scores)

        # Run Optuna to get best hyperparameters on current outer fold
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        logging.info(f"Best inner params (Fold {outer_fold+1}): {best_params}")

        # Train final model with best inner hyperparams on outer_train_val
        args.base_lr = best_params["base_lr"]
        args.weight_decay = best_params["weight_decay"]
        args.batch_size = best_params["batch_size"]
        args.pool_method = best_params["pool_method"]
        args.weight = best_params["weight"]
        args.heads_csa = best_params["heads_csa"]

        trainloader = DataLoader(outer_train_val, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
        testloader = DataLoader(outer_test, batch_size=4, shuffle=False, num_workers=2)

        model_instance = model_fn(args)

        if args.n_gpu > 1:
            model_instance = nn.DataParallel(model_instance)

        optimizer = optim.SGD(model_instance.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        writer = SummaryWriter(snapshot_path + f'/final_model_fold_{outer_fold}')
        iter_num = 0
        seg_iterator = tqdm(range(args.max_epochs_seg), ncols=70)
        cls_iterator = tqdm(range(args.max_epochs_cls), ncols=70)
        criterion_cls = nn.BCEWithLogitsLoss()

        train_segmentation(args, model_instance, trainloader, testloader, optimizer, scheduler, seg_iterator, args.weight,
                           writer, args.max_epochs_seg, snapshot_path, iter_num)
        # train_classification(args, model, trainloader, optimizer, scheduler, cls_iterator, args.base_lr, criterion_cls,
        #                      writer, args.max_epochs_cls, args.max_epochs_cls * len(trainloader), snapshot_path, iter_num)

        test_score = evaluate_dice(model_instance, testloader)
        logging.info(f"Outer Fold {outer_fold+1} test score: {test_score}")
        outer_scores.append(test_score)

        with open("best_params.txt", "a") as f:
            f.write(f"OUTER FOLD {outer_fold} best params: lr ({best_params["base_lr"]}), weight decay ({best_params["weight_decay"]}), batch size ({best_params["batch_size"]}), pool_method ({best_params["pool_method"]}), weight ({best_params["weight"]}), heads csa ({best_params["heads_csa"]})\n")
            f.write(f"Dice score: {test_score}\n")

    avg_score = sum(outer_scores) / len(outer_scores)
    logging.info(f"\n=== Final Nested CV Score: {avg_score} ===")

    return 'Training finished!'





def train_segmentation(args,model, trainloader, valloader, optimizer, scheduler, iterator, hard_weight,writer,max_epoch, snapshot_path, iter_num):
        #Freeze the classification head
        for param in model.classification_head.parameters():
            param.requires_grad=False

        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

        
        def evaluate_dice(model, valloader):
            model.eval()
            # val_transform = transforms.Compose(
            #     [MultiscaleGenerator(split = "val", output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]
            # )
            # val_dataset = MicroUS_dataset(base_dir=args.root_path_val, list_dir=args.list_dir, split="val", transform=val_transform)
            # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

            dices = []
            with torch.no_grad():
                for batch in valloader:
                    image = batch['image'].cuda()
                    label = batch['label'].cuda()

                    output, *_ = model(image)
                    pred = (torch.sigmoid(output.squeeze(1)) > 0.5).float()
        
                    for i in range(len(pred)):
                        if pred[i].sum(dim=(0,1)) == 0:  # No foreground in prediction
                            if label[i].sum(dim=(0,1)) == 0:
                                dices.append(1.0)
                            else:
                                dices.append(0.0)
                        else:
                            if label[i].sum(dim=(0,1)) == 0:
                                dices.append(0.0)
                            else:
                                dices.append(metric.binary.dc(pred[i].cpu().numpy(), label[i].cpu().numpy()))

            model.train()
            return np.mean(dices)
        
        with open("logit_logs.txt","w") as f:

            best_dice = 0
            counter = 0
            stop_patience = args.stop_patience

            # for epoch_num in iterator:
            for epoch_num in range(max_epoch):
                # transform = transforms.Compose(
                # [MultiscaleGenerator(split = "train", output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]
                # )
                # db = MicroUS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", transform=transform)
                # trainloader = DataLoader(db, batch_size=args.batch_size * args.n_gpu, shuffle=False, num_workers=8,
                #     worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id), drop_last=True)
                epoch_loss = []
                for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
                    transform_aug = A.Compose(
                        [
                            A.OneOf([
                                A.HorizontalFlip(p=1.0),
                                A.VerticalFlip(p=1.0)
                            ], p=0.5),
                            A.Rotate(limit=15, p=0.7)
                        ],
                        additional_targets={
                            'label': 'mask',
                            'non_expert': 'mask',
                        }
                    )

                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['non_expert'], sampled_batch['non_expert0'], sampled_batch['non_expert1'], sampled_batch['non_expert2']
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), non_expert_batch.cuda(), non_expert0_batch.cuda(), non_expert1_batch.cuda(), non_expert2_batch.cuda()
                    
                    # Calculate cls_label_batch
                    cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
                    outputs, out0, out1, out2, cls_output = model(image_batch)
                    outputs = torch.sigmoid(outputs).squeeze(dim=1)
                    out0 = torch.sigmoid(out0).squeeze(dim=1)
                    out1 = torch.sigmoid(out1).squeeze(dim=1)
                    out2 = torch.sigmoid(out2).squeeze(dim=1)


                    loss0 = attention_BCE_loss(hard_weight, label0_batch, out0, non_expert0_batch, ks=1)
                    loss1 = attention_BCE_loss(hard_weight, label1_batch, out1, non_expert1_batch, ks=2)
                    loss2 = attention_BCE_loss(hard_weight, label2_batch, out2, non_expert2_batch, ks=3)
                    loss3 = attention_BCE_loss(hard_weight, label_batch, outputs, non_expert_batch, ks=4)
                    #print(f"cls_loss:{cls_loss}")
                    cls_output = cls_output.squeeze(-1)

                    ######################################################################################
                     
                    image_batch_np = image_batch.cpu().numpy()
                    label_batch_np = label_batch.cpu().numpy()
                    label0_batch_np = label0_batch.cpu().numpy()
                    label1_batch_np = label1_batch.cpu().numpy()
                    label2_batch_np = label2_batch.cpu().numpy()
                    non_expert_batch_np = non_expert_batch.cpu().numpy()
                    non_expert0_batch_np = non_expert0_batch.cpu().numpy()
                    non_expert1_batch_np = non_expert1_batch.cpu().numpy()
                    non_expert2_batch_np = non_expert2_batch.cpu().numpy()

                    augmented_images = []
                    augmented_labels = []
                    augmented_label0 = []
                    augmented_label1 = []
                    augmented_label2 = []
                    augmented_non_expert = []
                    augmented_non_expert0 = []
                    augmented_non_expert1 = []
                    augmented_non_expert2 = []

                    for i in range(image_batch_np.shape[0]):
                        augmented = transform_aug(
                            image=image_batch_np[i, 0],  # assume shape (B, 1, H, W)
                            label=label_batch_np[i],
                            non_expert=non_expert_batch_np[i],
                        )
                        augmented0 = transform_aug(
                            image=cv2.resize(image_batch_np[i, 0], (28,28), interpolation=cv2.INTER_LINEAR),  # assume shape (B, 1, H, W)
                            label=label0_batch_np[i],
                            non_expert=non_expert0_batch_np[i],
                        )
                        augmented1 = transform_aug(
                            image=cv2.resize(image_batch_np[i, 0], (56,56), interpolation=cv2.INTER_LINEAR),  # assume shape (B, 1, H, W)
                            label=label1_batch_np[i],
                            non_expert=non_expert1_batch_np[i],
                        )
                        augmented2 = transform_aug(
                            image=cv2.resize(image_batch_np[i, 0], (112,112), interpolation=cv2.INTER_LINEAR),  # assume shape (B, 1, H, W)
                            label=label2_batch_np[i],
                            non_expert=non_expert2_batch_np[i],
                        )
                        
                        # Re-add channel dim if needed
                        augmented_images.append(augmented["image"][None])
                        augmented_labels.append(augmented["label"][None])
                        augmented_label0.append(augmented0["label"][None])
                        augmented_label1.append(augmented1["label"][None])
                        augmented_label2.append(augmented2["label"][None])
                        augmented_non_expert.append(augmented["non_expert"][None])
                        augmented_non_expert0.append(augmented0["non_expert"][None])
                        augmented_non_expert1.append(augmented1["non_expert"][None])
                        augmented_non_expert2.append(augmented2["non_expert"][None])

                    # Stack and convert back to tensors

                    image_batch_aug = torch.from_numpy(np.stack(augmented_images)).cuda()
                    label_batch_aug = torch.from_numpy(np.stack(augmented_labels).squeeze(1)).cuda()
                    label0_batch_aug = torch.from_numpy(np.stack(augmented_label0).squeeze(1)).cuda()
                    label1_batch_aug = torch.from_numpy(np.stack(augmented_label1).squeeze(1)).cuda()
                    label2_batch_aug = torch.from_numpy(np.stack(augmented_label2).squeeze(1)).cuda()
                    non_expert_batch_aug = torch.from_numpy(np.stack(augmented_non_expert).squeeze(1)).cuda()
                    non_expert0_batch_aug = torch.from_numpy(np.stack(augmented_non_expert0).squeeze(1)).cuda()
                    non_expert1_batch_aug = torch.from_numpy(np.stack(augmented_non_expert1).squeeze(1)).cuda()
                    non_expert2_batch_aug = torch.from_numpy(np.stack(augmented_non_expert2).squeeze(1)).cuda()

                     
                     
                                        
                    # Calculate cls_label_batch
                    cls_label_batch_aug = (label_batch_aug.sum(dim=(1,2)) > 0).float()
                    outputs_aug, out0_aug, out1_aug, out2_aug, cls_output_aug = model(image_batch_aug)
                    outputs_aug = torch.sigmoid(outputs_aug).squeeze(dim=1)
                    out0_aug = torch.sigmoid(out0_aug).squeeze(dim=1)
                    out1_aug = torch.sigmoid(out1_aug).squeeze(dim=1)
                    out2_aug = torch.sigmoid(out2_aug).squeeze(dim=1)

                    loss0_aug = attention_BCE_loss(hard_weight, label0_batch_aug, out0_aug, non_expert0_batch_aug, ks=1)
                    loss1_aug = attention_BCE_loss(hard_weight, label1_batch_aug, out1_aug, non_expert1_batch_aug, ks=2)
                    loss2_aug = attention_BCE_loss(hard_weight, label2_batch_aug, out2_aug, non_expert2_batch_aug, ks=3)
                    loss3_aug = attention_BCE_loss(hard_weight, label_batch_aug, outputs_aug, non_expert_batch_aug, ks=4)
                    #print(f"cls_loss:{cls_loss}")
                    cls_output_aug = cls_output_aug.squeeze(-1)


                    ######################################################################################
                    f.write(f"Cls_Label_batch_aug_val:  {cls_label_batch_aug}")
                    f.write(f"Cls_output_aug: {torch.sigmoid(cls_output_aug)}")
            
                    #For  classification
                    #loss = loss0 + loss1 + loss2 + loss3 + 0.065*cls_loss
                    #Standard
                    loss = (loss0 + loss1 + loss2 + loss3 + loss0_aug + loss1_aug + loss2_aug + loss3_aug) / 2

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    # for param_group in optimizer.param_groups:
                    #     param_group['lr'] = lr_

                    iter_num = iter_num + 1
                    epoch_loss.append(loss)
                    # writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)

                average_loss = sum(epoch_loss)/len(epoch_loss)
                logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))
                torch.save(model.state_dict(), os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth'))

                # --- Evaluate on validation set ---
                val_dice = evaluate_dice(model, valloader)

                scheduler.step(val_dice)
                writer.add_scalar('info/val_dice', val_dice, epoch_num)
                logging.info('Epoch %d : val_dice : %.4f' % (epoch_num, val_dice))
                # logging.info('Epoch %d : test_dice : %.4f' % (epoch_num, test_dice))
                print(f"Epoch {epoch_num} : val_dice = {val_dice:.4f}")

                print(f"VAL DICE: {val_dice}")
                print(f"BEST_DICE: {best_dice}")

                if val_dice > best_dice:
                    save_mode_path = os.path.join(snapshot_path, 'best_model' + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    iterator.close()
                    best_dice = val_dice
                    counter = 0
                else:
                    counter +=1
                
                if counter >= stop_patience:
                    logging.info(f"Early stopping triggered after {epoch_num + 1} epochs.")
                    break
                continue

def train_classification(args, model, trainloader, optimizer, scheduler, iterator, base_lr ,criterion_cls, writer, max_epoch, max_iterations, snapshot_path, iter_num):
    for param in model.parameters():
         param.requires_grad= False

    for param in model.classification_head.parameters():
            param.requires_grad=True

    def evaluate_dice(model, args):
        model.eval()
        val_transform = transforms.Compose(
            [MultiscaleGenerator(split = "train", output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]
        )
        val_dataset = MicroUS_dataset(base_dir=args.root_path_val, list_dir=args.list_dir, split="val", transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

        dices = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].cuda()
                label = batch['label'].cuda()

                output, *_ = model(image)
                pred = (torch.sigmoid(output.squeeze(1)) > 0.5).float()
        
                # print(f"PREDICTION: {np.unique(pred.cpu().numpy())}")
                # print(f"LABEL: {np.unique(label.cpu().numpy())}")

                for i in range(len(pred)):
                    if pred[i].sum(dim=(0,1)) == 0:  # No foreground in prediction
                        if label[i].sum(dim=(0,1)) == 0:
                            dices.append(1.0)
                        else:
                            dices.append(0.0)
                    else:
                        if label[i].sum(dim=(0,1)) == 0:
                            dices.append(0.0)
                        else:
                            dices.append(metric.binary.dc(pred[i].cpu().numpy(), label[i].cpu().numpy()))
                # print(f"LEN DICES: {len(dices)}")
                # print(f"DICES: {dices}")

                # intersection = (pred * label).sum(dim=(1, 2))
                # union = pred.sum(dim=(1, 2)) + label.sum(dim=(1, 2))
                # dice = (2.0 * intersection) / (union + 1e-5)

                # dices.extend(dice.cpu().numpy())

        model.train()
        return np.mean(dices)

    # def evaluate_dice_test(model, args):
    #     model.eval()
    #     val_transform = transforms.Compose(
    #         [MultiscaleGenerator(split = "test", output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]
    #     )
    #     val_dataset = MicroUS_dataset(base_dir=args.root_path_test, list_dir=args.list_dir, split="test", transform=val_transform)
    #     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    #     dices = []
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             image = batch['image'].cuda()
    #             label = batch['label'].cuda()

    #             output, *_ = model(image)
    #             pred = (torch.sigmoid(output.squeeze(1)) > 0.5).float()
        
    #             # print(f"PREDICTION: {np.unique(pred.cpu().numpy())}")
    #             # print(f"LABEL: {np.unique(label.cpu().numpy())}")

    #             for i in range(len(pred)):
    #                 if pred[i].sum(dim=(0,1)) == 0:  # No foreground in prediction
    #                     if label[i].sum(dim=(0,1)) == 0:
    #                         dices.append(1.0)
    #                     else:
    #                         dices.append(0.0)
    #                 else:
    #                     if label[i].sum(dim=(0,1)) == 0:
    #                         dices.append(0.0)
    #                     else:
    #                         dices.append(metric.binary.dc(pred[i].cpu().numpy(), label[i].cpu().numpy()))
    #             # print(f"LEN DICES: {len(dices)}")
    #             # print(f"DICES: {dices}")

    #             # intersection = (pred * label).sum(dim=(1, 2))
    #             # union = pred.sum(dim=(1, 2)) + label.sum(dim=(1, 2))
    #             # dice = (2.0 * intersection) / (union + 1e-5)

    #             # dices.extend(dice.cpu().numpy())

    #     model.train()
    #     return np.mean(dices)

    with open("logit_logs_classification.txt","w") as f:
            best_dice = 0
            counter = 0
            stop_patience = args.stop_patience


            for epoch_num in iterator:
                epoch_loss = []
                for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
                    transform_aug = A.Compose(
                        [
                            A.OneOf([
                                A.HorizontalFlip(p=1.0),
                                A.VerticalFlip(p=1.0)
                            ], p=0.5),
                            A.Rotate(limit=45, p=0.7)
                        ],
                    )
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['non_expert'], sampled_batch['non_expert0'], sampled_batch['non_expert1'], sampled_batch['non_expert2']
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), non_expert_batch.cuda(), non_expert0_batch.cuda(), non_expert1_batch.cuda(), non_expert2_batch.cuda()
                    # Calculate cls_label_batch
                    cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
                    outputs, out0, out1, out2, cls_output = model(image_batch)


                                        
                    cls_output = cls_output.squeeze(-1)            
                    cls_loss = criterion_cls(cls_output, cls_label_batch.float())

                    f.write(f"Cls_Label_batch_val:  {cls_label_batch}")
                    f.write(f"Cls_output: {torch.sigmoid(cls_output)}")
           
                    #print(f"cls_loss:{cls_loss}")
                    cls_output = cls_output.squeeze(-1)

                    ######################################################################################
                     
                    image_batch_np = image_batch.cpu().numpy()
                    label_batch_np = label_batch.cpu().numpy()
                    label0_batch_np = label0_batch.cpu().numpy()
                    label1_batch_np = label1_batch.cpu().numpy()
                    label2_batch_np = label2_batch.cpu().numpy()
                    non_expert_batch_np = non_expert_batch.cpu().numpy()
                    non_expert0_batch_np = non_expert0_batch.cpu().numpy()
                    non_expert1_batch_np = non_expert1_batch.cpu().numpy()
                    non_expert2_batch_np = non_expert2_batch.cpu().numpy()

                    augmented_images = []
                    augmented_labels = []
                    augmented_label0 = []
                    augmented_label1 = []
                    augmented_label2 = []
                    augmented_non_expert = []
                    augmented_non_expert0 = []
                    augmented_non_expert1 = []
                    augmented_non_expert2 = []

                    for i in range(image_batch_np.shape[0]):
                        augmented = transform_aug(
                            image=image_batch_np[i, 0],  # assume shape (B, 1, H, W)
                            label=label_batch_np[i],
                            non_expert=non_expert_batch_np[i],
                        )
                        augmented0 = transform_aug(
                            image=cv2.resize(image_batch_np[i, 0], (28,28), interpolation=cv2.INTER_LINEAR),  # assume shape (B, 1, H, W)
                            label=label0_batch_np[i],
                            non_expert=non_expert0_batch_np[i],
                        )
                        augmented1 = transform_aug(
                            image=cv2.resize(image_batch_np[i, 0], (56,56), interpolation=cv2.INTER_LINEAR),  # assume shape (B, 1, H, W)
                            label=label1_batch_np[i],
                            non_expert=non_expert1_batch_np[i],
                        )
                        augmented2 = transform_aug(
                            image=cv2.resize(image_batch_np[i, 0], (112,112), interpolation=cv2.INTER_LINEAR),  # assume shape (B, 1, H, W)
                            label=label2_batch_np[i],
                            non_expert=non_expert2_batch_np[i],
                        )
                        
                        # Re-add channel dim if needed
                        augmented_images.append(augmented["image"][None])
                        augmented_labels.append(augmented["label"][None])
                        augmented_label0.append(augmented0["label"][None])
                        augmented_label1.append(augmented1["label"][None])
                        augmented_label2.append(augmented2["label"][None])
                        augmented_non_expert.append(augmented["non_expert"][None])
                        augmented_non_expert0.append(augmented0["non_expert"][None])
                        augmented_non_expert1.append(augmented1["non_expert"][None])
                        augmented_non_expert2.append(augmented2["non_expert"][None])

                    # Stack and convert back to tensors

                    image_batch_aug = torch.from_numpy(np.stack(augmented_images)).cuda()
                    label_batch_aug = torch.from_numpy(np.stack(augmented_labels).squeeze(1)).cuda()
                    label0_batch_aug = torch.from_numpy(np.stack(augmented_label0).squeeze(1)).cuda()
                    label1_batch_aug = torch.from_numpy(np.stack(augmented_label1).squeeze(1)).cuda()
                    label2_batch_aug = torch.from_numpy(np.stack(augmented_label2).squeeze(1)).cuda()
                    non_expert_batch_aug = torch.from_numpy(np.stack(augmented_non_expert).squeeze(1)).cuda()
                    non_expert0_batch_aug = torch.from_numpy(np.stack(augmented_non_expert0).squeeze(1)).cuda()
                    non_expert1_batch_aug = torch.from_numpy(np.stack(augmented_non_expert1).squeeze(1)).cuda()
                    non_expert2_batch_aug = torch.from_numpy(np.stack(augmented_non_expert2).squeeze(1)).cuda()

                     
                     
                                        
                    # Calculate cls_label_batch
                    cls_label_batch_aug = (label_batch_aug.sum(dim=(1,2)) > 0).float()
                    outputs_aug, out0_aug, out1_aug, out2_aug, cls_output_aug = model(image_batch_aug)
                    outputs_aug = torch.sigmoid(outputs_aug).squeeze(dim=1)
                    out0_aug = torch.sigmoid(out0_aug).squeeze(dim=1)
                    out1_aug = torch.sigmoid(out1_aug).squeeze(dim=1)
                    out2_aug = torch.sigmoid(out2_aug).squeeze(dim=1)

                    #print(f"cls_loss:{cls_loss}")
                    cls_output_aug = cls_output_aug.squeeze(-1)
                    cls_loss_aug = criterion_cls(cls_output_aug, cls_label_batch_aug.float())



                    ######################################################################################
                    
            
                    #For classification
                    loss =0.5*cls_loss
                    loss_aug = 0.5*cls_loss_aug

                    loss = (loss + loss_aug)/2
                    

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    # for param_group in optimizer.param_groups:
                    #     param_group['lr'] = lr_

                    iter_num = iter_num + 1
                    epoch_loss.append(loss)


                    # writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)

                average_loss = sum(epoch_loss)/len(epoch_loss)

                val_dice = evaluate_dice(model, args)
                # test_dice = evaluate_dice_test(model, args)

                scheduler.step(val_dice)

                logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))
                logging.info('Epoch %d : val_dice : %.4f' % (epoch_num, val_dice))
                # logging.info('Epoch %d : test_dice : %.4f' % (epoch_num, test_dice))

                if val_dice > best_dice:
                    save_mode_path = os.path.join(snapshot_path, 'best_model_cls' + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    iterator.close()
                    best_dice = val_dice
                    counter = 0
                else:
                    counter +=1
                
                if counter >= stop_patience:
                    logging.info(f"Early stopping triggered after {epoch_num + 1} epochs.")
                    break
                continue