import uuid
from torchvision.utils import save_image
import torch
import torchvision
def log_epoch(writer, modus, epoch, loss,mse, psnr, org, rec, target, pred_img, mask_img):
    # log losses / metrics
    writer.add_scalar(modus + "/loss", loss, epoch)
    # writer.add_scalar(modus + "/bpppc", bpppc, epoch)
    writer.add_scalar(modus + "/mse", mse, epoch)
    writer.add_scalar(modus + "/psnr", psnr, epoch)
    # writer.add_scalar(modus + "/ssim", ssim, epoch)
    # writer.add_scalar(modus + "/sa", sa, epoch)
    # log images
    idx_c = [44, 29, 11]  # r, g, b channels
    # # log original image batch
    writer.add_images(f"{modus}/_org", org[:, idx_c, :, :], epoch, dataformats='NCHW')
    # # log reconstructed image batch
    writer.add_images(f"{modus}/_rec", pred_img[:, idx_c, :, :], epoch, dataformats='NCHW')
    writer.add_images(f"{modus}/_target", mask_img[:, idx_c, :, :], epoch, dataformats='NCHW')
    # img = torch.stack([zip(org[:, idx_c, :, :], rec[:, idx_c, :, :], target[:, idx_c, :, :])])
    # grid = torchvision.utils.make_grid(img, nrow=6)
    # # log reconstructed image batch
    # writer.add_images(f"{modus}/_rec", grid, epoch, dataformats='NCHW')
    # writer.add_images(f"{modus}/_rec", grid, epoch, dataformats='NCHW')
    # writer.add_images(f"{modus}/_rec", grid, epoch, dataformats='NCHW')
    # # log.logger.experiment.add_image("generated_images", grid, 0)
