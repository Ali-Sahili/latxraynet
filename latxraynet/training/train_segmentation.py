
import os
import torch
from tqdm import tqdm


from ..utils.seg_utils import dice_loss, dice_coeff, iou_score
from ..utils.visualize import plot_seg_results

#-------------------------------------------------------------------------
def train_seg(model, train_loader, val_loader, device, num_epochs=40, res_path=""):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()    

    best_dice = 0

    dice_list = []
    iou_list = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        # ----- TRAIN -----
        model.train()
        train_loss = 0

        for img, mask, _ in tqdm(train_loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)

            loss = bce(pred, mask) + dice_loss(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss/len(train_loader)

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0
        dice_total, iou_total = 0, 0

        with torch.no_grad():
            for img, mask, _ in val_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)

                loss = bce(pred, mask) + dice_loss(pred, mask)

                dice_total += dice_coeff(pred, mask).item()
                iou_total += iou_score(pred, mask).item()
                val_loss += loss.item()

        dice_avg = dice_total / len(val_loader)
        iou_avg = iou_total / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)

        iou_list.append(iou_avg)
        dice_list.append(dice_avg)
        train_loss_list.append(avg_loss)
        val_loss_list.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        print(f"             Val Dice: {dice_avg:.4f}, Val IoU: {iou_avg:.4f}")
        print(" ----------------------------------------------- ")
        if dice_avg > best_dice:
            best_dice = dice_avg
            torch.save(model.state_dict(), 
                os.path.join(res_path, "best_segmentation.pth"))
    
    plot_seg_results(train_loss_list, val_loss_list, dice_list, iou_list, 
        model_name = "basic_model_segmentation", res_path = res_path, save=True)