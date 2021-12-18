import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.iou_score import iou_score

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    criterion = torch.nn.CrossEntropyLoss()

    iou = 0
    accuracy = 0
    loss = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_img = mask_true
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            loss += criterion(mask_pred, mask_true_img).item()
            masks_pred_img = torch.argmax(mask_pred, dim=1)

            match = (masks_pred_img == mask_true_img).type(torch.bool).sum()
            shape_total = masks_pred_img.size(-2) * masks_pred_img.size(-1)
            accuracy += match / (shape_total * masks_pred_img.size(0))

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                iou += iou_score(mask_pred, mask_true, net.n_classes)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                mask_pred = mask_pred.permute(0, 3, 1, 2).float()
                iou += iou_score(mask_pred, mask_true, net.n_classes)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou
    return ((loss / num_val_batches), (accuracy/num_val_batches), (iou / num_val_batches))
