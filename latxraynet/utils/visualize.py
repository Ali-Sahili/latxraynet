
import os
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------
def plot_results(loss_list, val_loss_list, acc_list, auc_list, 
                    model_name = "", res_path = "", save=False, show=False):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid

    axs[0, 0].plot(loss_list)
    axs[0, 0].set_title(f"{model_name} - Training Loss")

    axs[0, 1].plot(val_loss_list)
    axs[0, 1].set_title(f"{model_name} - Validation Loss")

    axs[1, 0].plot(acc_list)
    axs[1, 0].set_title(f"{model_name} - Validation Accuracy")

    axs[1, 1].plot(auc_list)
    axs[1, 1].set_title(f"{model_name} - Validation AUC")

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(res_path, f"{model_name}_training_metrics.png"), 
                        dpi=300)
    if show:
        plt.show()

#-------------------------------------------------------------------------
def plot_seg_results(loss_list, val_loss_list, dice_list, iou_list, 
                model_name = "Segmentation", res_path = "", save=False, show=False):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid

    axs[0, 0].plot(loss_list)
    axs[0, 0].set_title(f"{model_name} - Training Loss")

    axs[0, 1].plot(val_loss_list)
    axs[0, 1].set_title(f"{model_name} - Validation Loss")

    axs[1, 0].plot(dice_list)
    axs[1, 0].set_title(f"{model_name} - Validation Dice")

    axs[1, 1].plot(iou_list)
    axs[1, 1].set_title(f"{model_name} - Validation IoU")

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(res_path, f"{model_name}_training_metrics.png"), 
                        dpi=300)
    if show:
        plt.show()

#-------------------------------------------------------------------------
def plot_metrics(acc_list, auc_list, sp_list, st_list, 
                model_name = "head_classifier", res_path = "", save=False, show=False):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid

    axs[0, 0].plot(acc_list)
    axs[0, 0].set_title(f"{model_name} - Accuracy")

    axs[0, 1].plot(auc_list)
    axs[0, 1].set_title(f"{model_name} - AUC")

    axs[1, 0].plot(sp_list)
    axs[1, 0].set_title(f"{model_name} - Specificity")

    axs[1, 1].plot(st_list)
    axs[1, 1].set_title(f"{model_name} - Sensitivity")

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(res_path, f"{model_name}_training_metrics.png"), 
                        dpi=300)
    if show:
        plt.show()