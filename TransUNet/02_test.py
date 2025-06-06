import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.Synapse_dataset import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/test_png', help='root dir for test data')
parser.add_argument('--dataset', type=str,
                    default='SSEA_ViTUnet', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int, 
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs_seg', type=int, 
                    default=30, help='maximum epoch number to train')
parser.add_argument('--max_epochs_cls', type=int, 
                    default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                     default=4, help='batch_size per gpu')
parser.add_argument('--is_savenii', 
                    action="store_false", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str,
                     default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  
                    default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  
                    default=0.01, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, 
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, 
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int, 
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                     default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, 
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--weight', type=int, 
                    default=4, help='weight for hard regions, default is 4')
args = parser.parse_args()


def inference(args, model, test_save_path):
    db_test = Synapse_dataset(args.root_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    result_list = []

    array = np.empty((len(db_test) + 2,7), dtype='U50')
    array[0,1] = "Dice"
    array[0,2] = "HD95"
    array[0,3] = "Jaccard"
    array[0,4] = "Specificity"
    array[0,5] = "Precision"
    array[0,6] = "Recall"
    test_images_counter=0
    for i_batch, sampled_batch in enumerate(testloader):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)
        if metric_i[0]!= 0.0:
            test_images_counter+=1
            metric_list += np.array(metric_i)
            result_list.append(metric_i[0])
            result_list.append(metric_i[1])
            result_list.append(metric_i[2])
            result_list.append(metric_i[3])
            result_list.append(metric_i[4])
            result_list.append(metric_i[5])
            #print(metric_list)
            logging.info('idx %d case %s dice %f mean_hd95 %f jc %f sp %f precision %f recall %f' % (i_batch, case_name, metric_i[0], metric_i[1], metric_i[2],metric_i[3],metric_i[4],metric_i[5]))

        array[i_batch+1,0] = case_name
        array[i_batch+1,1] = metric_i[0]
        array[i_batch+1,2] = metric_i[1]
        array[i_batch+1,3] = metric_i[2]
        array[i_batch+1,4] = metric_i[3]
        array[i_batch+1,5] = metric_i[4]
        array[i_batch+1,6] = metric_i[5]

    metric_list = metric_list / test_images_counter

    print(test_images_counter)
    print(len(db_test))

    #Popraviti
    #print(F"Result: {result_list}")
    print(f"Metric list: {metric_list[0]}")
    mean_dice = np.mean(metric_list[0])
    mean_hd95 = np.mean(metric_list[1])
    mean_jc = np.mean(metric_list[2])
    mean_sp =np.mean(metric_list[3])
    mean_precision = np.mean(metric_list[4])
    mean_recall =np.mean(metric_list[5])
    logging.info('Mean testing performance: dice : %f hd95 : %f jc : %f  sp : %f precision : %f recall : %f' % (mean_dice, mean_hd95, mean_jc, mean_sp, mean_precision, mean_recall))
    #logging.info(result_list)

    array[-1, 0] = "Average"
    array[-1, 1] = mean_dice
    array[-1, 2] = mean_hd95
    array[-1, 3] = mean_jc
    array[-1, 4] = mean_sp
    array[-1, 5] = mean_precision
    array[-1, 6] = mean_recall

    # Save csv file
    log_folder = './test_log/test_log_' + args.exp
    save_path = log_folder + '/' + 'test_result.csv'
    np.savetxt(save_path, array, delimiter=",", fmt='%s')

    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    args.is_pretrain = True
    args.exp = dataset_name
    snapshot_path = "./model/{}".format(args.exp)

    # Vit name
    snapshot_path += '_' + args.vit_name
    
    snapshot_path = snapshot_path + '_img'+ str(args.img_size)
    # max epoch
    snapshot_path = snapshot_path + '_seg_epo' + str(args.max_epochs_seg) if args.max_epochs_seg != 15 else snapshot_path
    snapshot_path = snapshot_path + '_cls_epo' + str(args.max_epochs_cls) if args.max_epochs_cls != 15 else snapshot_path

    # batch size
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    # base learning rate
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    config_vit.batch_size=1

    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    # load model
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
   
    #snapshot = os.path.join(snapshot_path, 'cls_epoch_'+f"{i}"+'.pth')
    snapshot = os.path.join(snapshot_path, 'best_seg.pth')
    #snapshot = os.path.join(snapshot_path, 'best_cls.pth')
    #snapshot = os.path.join(snapshot_path, 'epoch_'+str(args.max_epochs-1)+'.pth')
    print('The testing model is load from:', snapshot)
    net.load_state_dict(torch.load(snapshot, weights_only=True))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    logging.info(f"Model: {snapshot}")

    if args.is_savenii==1:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)
