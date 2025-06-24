import os
import random
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import copy

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer

#from trainer_optuna import trainer, objective
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/train_png', help='root dir for training data')
parser.add_argument('--unlabelled_data_path', type=str,
                    default='./data/unlabelled_img', help='root dir for training data')
parser.add_argument('--validation_path', type=str,
                    default='./data/validation_png', help='root dir for validaiton data')
parser.add_argument('--list_dir', type=str,
                    default='lists', help='list dir')
parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--warmup_epochs', type=int,
                    default=6, help='number of supervised epochs')
parser.add_argument('--max_epochs_seg', type=int,
                    default=20, help='maximum epoch number to train segmentation')
parser.add_argument('--max_epochs_cls', type=int,
                    default=20, help='maximum epoch number to train classification')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, 
                    default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  
                    default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  
                    default=0.04, help='segmentation network learning rate at the start')
parser.add_argument('--img_size', type=int,
                    default=224, help='input image size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=2, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--weight', type=int,
                    default=8, help='weight for hard regions, default is 4')                
args = parser.parse_args()


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

    args.is_pretrain = True
    args.exp = "SSEA_ViTUnet" 
    snapshot_path = f"./model/{args.exp}"

    snapshot_path = snapshot_path + "_" + args.vit_name + "_img" + str(args.img_size) + "_seg_epo"  + str(args.max_epochs_seg) + "_cls_epo" + str(args.max_epochs_cls) + "_bs" + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.batch_size=args.batch_size
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    #For full training
    config_vit.pretrained_path="./model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz"
    # #load model as student network
    student_net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    student_net.load_from(weights=np.load(config_vit.pretrained_path))

    #Setup the teacher network
    teacher_net = copy.deepcopy(student_net)
    teacher_net.eval()  #Set the teacher model to eval beacause it is not trained directly

    #For classification
    # MODEL_PATH="model/CRTA_Segmentation_R50-ViT-B_16_img224_seg_epo30_cls_epo30_bs4/seg_epoch12.pth"
    # student_net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # student_net.load_state_dict(torch.load(MODEL_PATH))
 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_net = student_net.to(device)  # Move the model to the correct device
   
    dataset_name = "SSEA_ViTUnet"

    # trainer_ = {'SSEA_ViTUnet': trainer}
    # trainer_[dataset_name](args, student_net, teacher_net, snapshot_path)

    import optuna

    # # Add this after model initialization
    # def optuna_objective(trial):
    #     return objective(args, student_net, teacher_net, snapshot_path, trial)

    # study = optuna.create_study(
    #     direction='maximize',
    #     sampler=optuna.samplers.TPESampler(),
    #     pruner=optuna.pruners.MedianPruner()
    # )
    # study.optimize(optuna_objective, n_trials=50)  # Run 50 trials


    #For optuna
    trainer_ = {'SSEA_ViTUnet': trainer}
    trainer_[dataset_name](args, student_net, teacher_net, snapshot_path)

    