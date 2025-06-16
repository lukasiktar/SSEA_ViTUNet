import numpy as np
import torch
from medpy import metric
import cv2
import torch.nn as nn
import SimpleITK as sitk
np.bool = np.bool_


class AreaWeightedLoss(nn.Module):
    def __init__(self, 
                 area_threshold=0.25, 
                 high_weight=1.5, 
                 low_weight=1.0, 
                 bce_weight=0.5, 
                 dice_weight=0.5):
        super(AreaWeightedLoss, self).__init__()
        self.area_threshold = area_threshold
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, labels):
        """
        preds: (B, H, W) - raw logits
        labels: (B, H, W) - binary ground truth
        """
        B = preds.size(0)

        # Ensure float type for loss computation
        labels = labels.float()

        # BCE Loss
        bce_raw = self.bce_loss_fn(preds, labels)  # (B, H, W)
        bce_per_sample = bce_raw.view(B, -1).mean(dim=1)  # (B,)

        # Dice Loss
        dice_per_sample = self.dice_loss(preds, labels)  # (B,)

        # Combine BCE and Dice losses
        hybrid_loss = self.bce_weight * bce_per_sample + self.dice_weight * dice_per_sample  # (B,)

        # Compute mask area for each sample
        area = labels.view(B, -1).mean(dim=1)  # (B,)
        weights = torch.where(area < self.area_threshold,
                            torch.tensor(self.high_weight, device=labels.device),
                            torch.tensor(self.low_weight, device=labels.device))

        # Weighted average
        weighted_loss = (weights * hybrid_loss).mean()
        return weighted_loss

    def dice_loss(self, preds, labels, epsilon=1e-6):
        preds = torch.sigmoid(preds)
        B = preds.size(0)
        preds_flat = preds.view(B, -1)
        labels_flat = labels.view(B, -1)

        intersection = (preds_flat * labels_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + labels_flat.sum(dim=1)
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return 1 - dice  





# Annotation-guided binary cross entropy loss (AG-BCE)
def attention_BCE_loss(h_W, y_true, y_pred, y_std, ks = 5):
    number_of_pixels = y_true.shape[0]*y_true.shape[1]*y_true.shape[2]

    y_true_np = y_true.cpu().detach().numpy()
    y_std_np = y_std.cpu().detach().numpy()


    hard = cv2.bitwise_xor(y_true_np, y_std_np)
    hard = hard.astype(np.uint8)
    
    # Apply dilation operation to hard regions
    kernel_size = ks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(hard.shape[0]):
        hard[i] = cv2.dilate(hard[i], kernel)
    hard = hard.astype(np.float32)

    easy = abs(hard-1)
    hard = torch.tensor(hard).cuda()
    easy = torch.tensor(easy).cuda()

    epsilon = 0.000001
    beta = 0.5

    loss = -beta*torch.mul(y_true,torch.log(y_pred + epsilon)) - (1.0 - beta)*torch.mul(1.0-y_true,torch.log(1.0 - y_pred + epsilon))
    hard_loss = torch.sum(torch.mul(loss,hard))
    easy_loss = torch.sum(torch.mul(loss,easy))

    LOSS = ((1/(1+h_W))*easy_loss + (h_W/(1+h_W))*hard_loss)/(number_of_pixels)

    return LOSS


def calculate_metric_percase(pred, gt, spacing):
    pred = np.array(pred)
    gt = np.array(gt)
    pred[pred > 0] = 1  
    gt[gt > 0] = 1  
   
    if np.sum(pred) == 0:  # No foreground in prediction
        if np.sum(gt) == 0:
            hd95 = float(0.0)  
        else:
            
            hd95 = float(1.0)
    else:
        if np.sum(gt) == 0:
            hd95 = float(0.0)
        else:
            hd95 = metric.binary.hd95(pred, gt)

    dice = metric.binary.dc(pred, gt) if np.sum(gt) > 0 else 0  
    jc = metric.binary.jc(pred, gt) if np.sum(gt) > 0 else 0
    sp = metric.binary.specificity(pred, gt) if np.sum(gt) > 0 else 0
    precision = metric.binary.precision(pred,gt) if np.sum(gt) > 0 else 0
    recall = metric.binary.recall(pred,gt) if np.sum(gt) > 0 else 0
    
    return dice, hd95, jc, sp, precision, recall


def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, classification=True):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    predictions =[]
    labels=[]
    
    slice=image
    x, y = slice.shape[0], slice.shape[1]
    if x != patch_size[0] or y != patch_size[1]:
        slice = cv2.resize(slice, patch_size, interpolation = cv2.INTER_NEAREST)
   

    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs, _, _, _, cls_output = net(input)
        out = torch.sigmoid(outputs).squeeze()
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)  
        else:
            pred = out
        if classification == True:
            if torch.sigmoid(cls_output) < 0.9:
                pred = np.zeros_like(image) 
                print("Empty")

        a = 1.0*(pred>0.5)
        pred_uint8 = (a * 255).astype(np.uint8)
      
        slice = cv2.resize(slice, (y, x))
        label=label.astype(np.uint8)
        label_contours, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(pred_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        orig_slice = cv2.drawContours(slice, contours, 0, (255,255,255), 4)
        orig_slice = cv2.drawContours(slice, label_contours, 0, (255,0,0), 4)
        cv2.imshow("pred",orig_slice)
        cv2.waitKey(10)
        prediction = a.astype(np.uint8)
        
    
    metric_list = []
    # In-plane spacing is 0.033586mm*0.033586mm
    if classes == 1:
        metric_list=calculate_metric_percase(prediction, label, 0.033586)
    return metric_list

