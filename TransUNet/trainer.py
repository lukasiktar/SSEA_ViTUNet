import os
import sys
import logging
import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import attention_BCE_loss, AreaWeightedLoss, test_single_volume


def trainer(args, model, snapshot_path):
    from datasets.Synapse_dataset import MultiscaleGenerator, Synapse_dataset

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    #num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    hard_weight = args.weight

    db_train = Synapse_dataset(dataset_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [MultiscaleGenerator(output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epochs_seg = args.max_epochs_seg
    max_iterations = args.max_epochs_seg * len(trainloader)  
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    #Segmentation training parameters
    seg_iterator = tqdm(range(max_epochs_seg), ncols=70)
    criterion = AreaWeightedLoss(area_threshold=0.4,high_weight=0.12,low_weight=0.001,bce_weight=0.6,dice_weight=0.6)

    #Classification training parameters
    criterion_cls=nn.BCEWithLogitsLoss()
    max_epoch_cls = args.max_epochs_cls
    cls_iterator=tqdm(range(max_epoch_cls), ncols=70)

    #Validation setup
    db_validation=Synapse_dataset(dataset_dir=args.validation_path, split="validation", list_dir=args.list_dir)
    validationloader=DataLoader(db_validation, batch_size=1, shuffle=False, num_workers=1)
   
    model=train_segmentation(args,model, trainloader, optimizer, db_validation, validationloader, seg_iterator, base_lr, hard_weight,writer,max_epochs_seg, max_iterations, snapshot_path, iter_num, criterion)

    train_classification(args, model, trainloader, optimizer, db_validation, validationloader, cls_iterator, base_lr ,criterion_cls, writer, max_epoch_cls, max_iterations, snapshot_path, iter_num)          

    writer.close()
    logging.shutdown()

    return "Training Finished!"

def validation(args, model, db_validation, validationloader, max_dice, snapshot_path, best_model, segmentation=True):
    model.eval()
    metric_list = 0.0
    result_list = []

    array = np.empty((len(db_validation) + 2,7), dtype='U50')
    array[0,1] = "Dice"
    array[0,2] = "HD95"
    array[0,3] = "Jaccard"
    array[0,4] = "Specificity"
    array[0,5] = "Precision"
    array[0,6] = "Recall"
    test_images_counter=0

    for i_batch, sampled_batch in enumerate(validationloader):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        test_save_path = "./predictions"
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],test_save_path=test_save_path, case=case_name, classification=False)
        if metric_i[0]!= 0.0:
            test_images_counter+=1
            metric_list += np.array(metric_i)
            result_list.append(metric_i[0])
            result_list.append(metric_i[1])
            result_list.append(metric_i[2])
            result_list.append(metric_i[3])
            result_list.append(metric_i[4])
            result_list.append(metric_i[5])
        array[i_batch+1,0] = case_name
        array[i_batch+1,1] = metric_i[0]
        array[i_batch+1,2] = metric_i[1]
        array[i_batch+1,3] = metric_i[2]
        array[i_batch+1,4] = metric_i[3]
        array[i_batch+1,5] = metric_i[4]
        array[i_batch+1,6] = metric_i[5]
        #print(f"dice: {metric_i[0]}")
    try:
        metric_list = metric_list / test_images_counter

        mean_dice = np.mean(metric_list[0])
        mean_hd95 = np.mean(metric_list[1])
        mean_jc = np.mean(metric_list[2])
        mean_sp =np.mean(metric_list[3])
        mean_precision = np.mean(metric_list[4])
        mean_recall =np.mean(metric_list[5])
        logging.info('Mean validation performance: dice : %f hd95 : %f jc : %f  sp : %f precision : %f recall : %f' % (mean_dice, mean_hd95, mean_jc, mean_sp, mean_precision, mean_recall))
    except:
           mean_dice=0.0

    best_model=best_model
    if mean_dice > max_dice:
        max_dice=mean_dice
        best_model=model
        if segmentation:
            save_mode_path = os.path.join(snapshot_path, 'best_seg'+ '.pth')
        else:
            save_mode_path = os.path.join(snapshot_path, 'best_cls'+ '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        logging.info("mean dice of the best model is {}".format(max_dice))
    best_model.train()
    return max_dice, best_model

def train_segmentation(args,model, trainloader, optimizer, db_validation, validationloader,  iterator, base_lr, hard_weight,writer,max_epoch, max_iterations, snapshot_path, iter_num, criterion):
        #Freeze the classification head
        for param in model.classification_head.parameters():
            param.requires_grad=False

        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

        max_dice=0.0
        #Initialize the first model as the best model at the start of the training
        best_model=model
        for epoch_num in iterator:
            epoch_loss = []
            for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
                image_batch, label_batch, label0_batch, label1_batch, label2_batch, preannotation_batch, preannotation0_batch, preannotation1_batch, preannotation2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['preannotation'], sampled_batch['preannotation0'], sampled_batch['preannotation1'], sampled_batch['preannotation2']
                image_batch, label_batch, label0_batch, label1_batch, label2_batch, preannotation_batch, preannotation0_batch, preannotation1_batch, preannotation2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), preannotation_batch.cuda(), preannotation0_batch.cuda(), preannotation1_batch.cuda(), preannotation2_batch.cuda()
                
                #Augmentation process
                B=image_batch.size(0)
                aug_image_batch=[]
                aug_label_batch=[]
                aug_label0_batch=[]
                aug_label1_batch=[]
                aug_label2_batch=[] 
                aug_preannotation_batch=[] 
                aug_preannotation0_batch=[]
                aug_preannotation1_batch=[]
                aug_preannotation2_batch=[]

                for i in range(B):
                        image=image_batch[i]
                        label=label_batch[i]
                        label0=label0_batch[i]
                        label1=label1_batch[i]
                        label2=label2_batch[i]
                        preannotation=preannotation_batch[i]
                        preannotation0=preannotation0_batch[i]
                        preannotation1=preannotation1_batch[i]
                        preannotation2=preannotation2_batch[i] 

                        aug_image, aug_label, aug_label0, aug_label1, aug_label2, aug_preannotation, aug_preannotation0, aug_preannotation1, aug_preannotation2 = augment(image, label, label0, label1, label2, preannotation, preannotation0, preannotation1, preannotation2)
                        aug_image_batch.append(aug_image)
                        aug_label_batch.append(aug_label)
                        aug_label0_batch.append(aug_label0)
                        aug_label1_batch.append(aug_label1)
                        aug_label2_batch.append(aug_label2)
                        aug_preannotation_batch.append(aug_preannotation)
                        aug_preannotation0_batch.append(aug_preannotation0)
                        aug_preannotation1_batch.append(aug_preannotation1)
                        aug_preannotation2_batch.append(aug_preannotation2)
                aug_image_batch=torch.stack(aug_image_batch)
                aug_label_batch=torch.stack(aug_label_batch)
                aug_label0_batch=torch.stack(aug_label0_batch)
                aug_label1_batch=torch.stack(aug_label1_batch)
                aug_label2_batch=torch.stack(aug_label2_batch)
                aug_preannotation_batch=torch.stack(aug_preannotation_batch)
                aug_preannotation0_batch=torch.stack(aug_preannotation0_batch)
                aug_preannotation1_batch=torch.stack(aug_preannotation1_batch)
                aug_preannotation2_batch=torch.stack(aug_preannotation2_batch)


                # # Calculate cls_label_batch
                # cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
                # outputs, out0, out1, out2, cls_output = model(image_batch)
                # outputs = torch.sigmoid(outputs).squeeze(dim=1)
                # out0 = torch.sigmoid(out0).squeeze(dim=1)
                # out1 = torch.sigmoid(out1).squeeze(dim=1)
                # out2 = torch.sigmoid(out2).squeeze(dim=1)

                # loss0 = attention_BCE_loss(hard_weight, label0_batch, out0, preannotation0_batch, ks=1)
                # loss1 = attention_BCE_loss(hard_weight, label1_batch, out1, preannotation1_batch, ks=2)
                # loss2 = attention_BCE_loss(hard_weight, label2_batch, out2, preannotation2_batch, ks=3)
                # loss3 = attention_BCE_loss(hard_weight, label_batch, outputs, preannotation_batch, ks=4)
                # #print(f"cls_loss:{cls_loss}")
                # cls_output = cls_output.squeeze(-1)

                # loss4 = criterion(outputs, label_batch)

                # Calculate cls_label_batch
                cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
                outputs, out0, out1, out2, cls_output = model(aug_image_batch)
                outputs = torch.sigmoid(outputs).squeeze(dim=1)
                out0 = torch.sigmoid(out0).squeeze(dim=1)
                out1 = torch.sigmoid(out1).squeeze(dim=1)
                out2 = torch.sigmoid(out2).squeeze(dim=1)

                loss0 = attention_BCE_loss(hard_weight, aug_label0_batch, out0, aug_preannotation0_batch, ks=1)
                loss1 = attention_BCE_loss(hard_weight, aug_label1_batch, out1, aug_preannotation1_batch, ks=2)
                loss2 = attention_BCE_loss(hard_weight, aug_label2_batch, out2, aug_preannotation2_batch, ks=3)
                loss3 = attention_BCE_loss(hard_weight, aug_label_batch, outputs, aug_preannotation_batch, ks=4)
                cls_output = cls_output.squeeze(-1)

                #AreaWeighted loss
                loss4 = criterion(outputs, aug_label_batch)
                #Standard
                loss = loss0 + loss1 + loss2 + loss3 + loss4
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                epoch_loss.append(loss)
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)

            average_loss = sum(epoch_loss)/len(epoch_loss)
            logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))

            iterator.close()
            save_mode_path = os.path.join(snapshot_path, 'seg_epoch'+ str(epoch_num) +'.pth')
            torch.save(best_model.state_dict(), save_mode_path)

            max_dice, best_model =validation(args, model, db_validation, validationloader, max_dice,snapshot_path, best_model)

            if epoch_num >= max_epoch - 1:
                logging.info('Epoch %d : The best segmentation model saved as: best_seg.pth')
                save_mode_path = os.path.join(snapshot_path, 'best_seg'+ '.pth')
                torch.save(best_model.state_dict(), save_mode_path)
                model=best_model
                return best_model
            continue

def train_classification(args, model, trainloader, optimizer,  db_validation, validationloader, iterator, base_lr ,criterion_cls, writer, max_epoch, max_iterations, snapshot_path, iter_num):
    best_model=model
    model.train()
    for param in model.parameters():
         param.requires_grad= False
        

    for param in model.classification_head.parameters():
            param.requires_grad=True
    for param in model.segmentation_head.parameters():
        param.requires_grad = False

    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         module.eval()

    max_dice=0.0

    for epoch_num in iterator:
        epoch_loss = []
        for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
            image_batch, label_batch, label0_batch, label1_batch, label2_batch, preannotation_batch, preannotation0_batch, preannotation1_batch, preannotation2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['preannotation'], sampled_batch['preannotation0'], sampled_batch['preannotation1'], sampled_batch['preannotation2']
            image_batch, label_batch, label0_batch, label1_batch, label2_batch, preannotation_batch, preannotation0_batch, preannotation1_batch, preannotation2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), preannotation_batch.cuda(), preannotation0_batch.cuda(), preannotation1_batch.cuda(), preannotation2_batch.cuda()
            # Calculate cls_label_batch
            cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
            outputs, out0, out1, out2, cls_output = model(image_batch)
                                
            cls_output = cls_output.squeeze(-1)            
            cls_loss = criterion_cls(cls_output, cls_label_batch.float())

            cls_output = cls_output.squeeze(-1)
            
            #For classification
            loss =0.5*cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            lr_ = base_lr * max(0, 1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            epoch_loss.append(loss)
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

        average_loss = sum(epoch_loss)/len(epoch_loss)
        logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))

        save_mode_path = os.path.join(snapshot_path, 'cls_epoch'+ str(epoch_num) +'.pth')
        torch.save(best_model.state_dict(), save_mode_path)

        max_dice, best_model =validation(args, model, db_validation, validationloader, max_dice,snapshot_path, best_model, segmentation=False)

        iterator.close()
        continue

def augment(image, label, label0, label1, label2, preannotation, preannotation0, preannotation1, preannotation2):
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)
        label0 = TF.hflip(label0)
        label1 = TF.hflip(label1)
        label2 = TF.hflip(label2)
        preannotation = TF.hflip(preannotation)
        preannotation0 = TF.hflip(preannotation0)
        preannotation1 = TF.hflip(preannotation1)
        preannotation2 = TF.hflip(preannotation2)

    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-8, 8)
        image = TF.rotate(image, angle)
        label = TF.rotate(label.unsqueeze(0), angle).squeeze(0)
        label0 = TF.rotate(label0.unsqueeze(0), angle).squeeze(0)
        label1 = TF.rotate(label1.unsqueeze(0), angle).squeeze(0)
        label2 = TF.rotate(label2.unsqueeze(0), angle).squeeze(0)
        preannotation = TF.rotate(preannotation.unsqueeze(0), angle).squeeze(0)
        preannotation0 = TF.rotate(preannotation0.unsqueeze(0), angle).squeeze(0)
        preannotation1 = TF.rotate(preannotation1.unsqueeze(0), angle).squeeze(0)
        preannotation2 = TF.rotate(preannotation2.unsqueeze(0), angle).squeeze(0)
        
    return image, label, label0, label1, label2, preannotation, preannotation0, preannotation1, preannotation2