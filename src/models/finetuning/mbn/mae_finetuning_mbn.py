import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

import timm
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from torchvision.transforms import RandomCrop

from .magicbathynet_unet import UNet_bathy
from src.utils.finetuning_utils import calculate_metrics
from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, depth, mask):
        mse_loss = nn.MSELoss(reduction='none')

        loss = mse_loss(output, depth)
        loss = (loss * mask.float()).sum()

        non_zero_elements = mask.sum()
        rmse_loss_val = torch.sqrt(loss / non_zero_elements)

        return rmse_loss_val

class MAEFineTuning(pl.LightningModule):
    def __init__(self, src_channels=3, mask_ratio=0.5,pretrained_model=None,location="agia_napa",
                 full_finetune=False, random=False, ssl=False, model_type="mae"):

        super().__init__()
        self.writer = SummaryWriter()
        self.save_hyperparameters()

        self.run_dir = None
        self.base_dir = "bathymetry_results"
        self.pretrained_model = pretrained_model
        self.model_type = model_type
        self.base_lr = 0.0001

        self.src_channels = src_channels
        self.mask_ratio = mask_ratio
        self.location = location
        self.norm_param_depth = NORM_PARAM_DEPTH[location]
        self.norm_param = np.load(NORM_PARAM_PATHS[location])
        self.crop_size = MODEL_CONFIG["crop_size"]
        self.window_size = MODEL_CONFIG["window_size"]
        self.stride = MODEL_CONFIG["stride"]
        self.full_finetune = full_finetune

        self.projection_head = UNet_bathy(in_channels=3, out_channels=1, model_type=self.model_type, full_finetune=self.full_finetune)
        self.criterion = CustomLoss()

        self.total_train_loss = 0.0
        self.train_batch_count = 0
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []
        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []
        self.test_image_count = 0

        if self.full_finetune:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, images, embedding):
        processed_embedding = embedding

        if self.full_finetune:
            if self.model_type == "mae":
                print("embedding before",embedding.shape)
                embedding = embedding.squeeze(1)
                print("embedding after squeeze",embedding.shape)
                processed_embedding = self.pretrained_model.forward_encoder(embedding)
                print("processed_embedding",processed_embedding.shape)
            elif self.model_type in ["moco", "geo_aware","ocean_aware"]:
                print("embedding shape:", embedding.shape)
                embedding = embedding.squeeze(0)
                processed_embedding = self.pretrained_model.backbone(embedding).flatten(start_dim=1)

        return self.projection_head(processed_embedding, images)

    def training_step(self, batch,batch_idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dir = "training_results"
        size=(256, 256)
        data, target, embedding = batch
        data, target,embedding = Variable(data.to(device)), Variable(target.to(device)), Variable(embedding.to(device))

        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(0), size=size, mode='nearest')
        data_size = data.size()[2:]

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
            data_transform = RandomCrop(size=self.crop_size)
            target_transform = RandomCrop(size=self.crop_size)

            data = data_transform(data)
            target = target_transform(target)

        target_mask = (target.cpu().numpy() != 0).astype(np.float32)
        target_mask = torch.from_numpy(target_mask)
        target_mask = target_mask.reshape(self.crop_size, self.crop_size)
        target_mask = target_mask.to(device)

        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask)
        data_mask = data_mask.to(device)

        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float()
        if torch.sum(combined_mask) == 0:
            return None

        data = torch.clamp(data, min=0, max=1)

        output = self(data.float(),embedding.float())
        output = output.to(self.device)

        loss = self.criterion(output, target, combined_mask)

        rgb = np.asarray(np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='float32')
        pred = output.data.cpu().numpy()[0]
        gt = target.data.cpu().numpy()[0]
        masked_pred = pred * combined_mask.cpu().numpy()
        masked_gt = gt * combined_mask.cpu().numpy()

        if batch_idx == 0:
            self.log_images(
                rgb,
                masked_pred[0,:,:],
                gt[0,:,:],
                train_dir
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel())

        self.log('train_rmse_step', (rmse * -self.norm_param_depth), on_step=True)
        self.log('train_mae_step', (mae * -self.norm_param_depth), on_step=True)
        self.log('train_std_dev_step', (std_dev * -self.norm_param_depth), on_step=True)

        self.epoch_rmse_list.append(rmse * -self.norm_param_depth)
        self.epoch_mae_list.append(mae * -self.norm_param_depth)
        self.epoch_std_dev_list.append(std_dev * -self.norm_param_depth)

        print('Mean RMSE (per image):', rmse * -self.norm_param_depth)
        print('Mean MAE (per image):', mae * -self.norm_param_depth)
        print('Mean Std Dev (per image):', std_dev * -self.norm_param_depth)

        self.total_train_loss += loss.item()
        self.train_batch_count += 1

        return loss

    def validation_step(self, batch, batch_idx):
        val_dir = "validation_results"
        data, target, embedding = batch

        data, target,embedding = data.to(self.device), target.to(self.device), embedding.to(self.device)
        device = self.device
        size=(256, 256)
        data, target, embedding = batch
        data, target,embedding = Variable(data.to(device)), Variable(target.to(device)), Variable(embedding.to(device))

        data = F.interpolate(data, size=size, mode='nearest')
        target = F.interpolate(target.unsqueeze(0), size=size, mode='nearest')
        data_size = data.size()[2:]

        if data_size[0] > self.crop_size and data_size[1] > self.crop_size:
            data_transform = RandomCrop(size=self.crop_size)
            target_transform = RandomCrop(size=self.crop_size)

            data = data_transform(data)
            target = target_transform(target)

        target_mask = (target.cpu().numpy() != 0).astype(np.float32)
        target_mask = torch.from_numpy(target_mask)
        target_mask = target_mask.reshape(self.crop_size, self.crop_size)
        target_mask = target_mask.to(device)

        data_mask = (data.cpu().numpy() != 0).astype(np.float32)
        data_mask = np.mean(data_mask, axis=1)
        data_mask = torch.from_numpy(data_mask)
        data_mask = data_mask.to(device)

        combined_mask = target_mask * data_mask
        combined_mask = (combined_mask >= 0.5).float()
        if torch.sum(combined_mask) == 0:
            return None

        data = torch.clamp(data, min=0, max=1)

        output = self(data.float(),embedding.float())
        output = output.to(self.device)

        val_loss = self.criterion(output, target, combined_mask)

        rgb = np.asarray(np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='float32')
        pred = output.data.cpu().numpy()[0]
        gt = target.data.cpu().numpy()[0]
        masked_pred = pred * combined_mask.cpu().numpy()
        masked_gt = gt * combined_mask.cpu().numpy()

        if batch_idx == 0:
            self.log_images(
                rgb,
                masked_pred[0,:,:],
                gt[0,:,:],
                val_dir
            )

        rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt.ravel())

        self.log('val_rmse', (rmse * -self.norm_param_depth), on_step=True)
        self.log('val_mae', (mae * -self.norm_param_depth), on_step=True)
        self.log('val_std_dev', (std_dev * -self.norm_param_depth), on_step=True)

        self.val_rmse_list.append(rmse * -self.norm_param_depth)
        self.val_mae_list.append(mae * -self.norm_param_depth)
        self.val_std_dev_list.append(std_dev * -self.norm_param_depth)

        print('Mean RMSE (per image):', rmse * -self.norm_param_depth)
        print('Mean MAE (per image):', mae * -self.norm_param_depth)
        print('Mean Std Dev (per image):', std_dev * -self.norm_param_depth)

        self.total_val_loss += val_loss.item()
        self.val_batch_count += 1
        return val_loss

    def test_step(self, batch, batch_idx):
        test_dir = "test_results"
        test_data_batch, targets_batch, embeddings_batch = batch

        self.crop_size = 256
        pad_size = 32
        ratio = self.crop_size / self.window_size[0]

        for img, gt,gt_e, embedding in zip(test_data_batch, targets_batch,targets_batch, embeddings_batch):
            img = img.cpu()
            gt = gt.cpu()
            gt_e = gt_e.cpu()
            embedding = embedding.unsqueeze(0)

            img = scipy.ndimage.zoom(img, (1,ratio, ratio), order=1)
            gt = scipy.ndimage.zoom(gt, (ratio, ratio), order=1)
            gt_e = scipy.ndimage.zoom(gt_e, (ratio, ratio), order=1)

            img = torch.from_numpy(img)
            img = torch.clamp(img, min=0, max=1)

            gt = torch.from_numpy(gt).float()
            gt_e = torch.from_numpy(gt_e).float()

            with torch.no_grad():
                img = img.float()
                img = img.unsqueeze(0)
                print("self.full_finetune", self.full_finetune)
                outs = self(img, embedding)
                pred = outs.data.cpu().numpy().squeeze()

            gt_mask = (gt_e != 0)
            gt_mask = gt_mask.unsqueeze(0)
            gt_mask = gt_mask.reshape(self.crop_size, self.crop_size)
            gt_mask = gt_mask.to(self.device)

            img_mask = (img != 0).float()
            img_mask = img_mask.to(self.device)

        
            print("img_mask shape:", img_mask.shape)
            print("gt_mask shape:", gt_mask.shape)
            combined_mask = img_mask*gt_mask

            print("combined_mask shape:", combined_mask.shape)

            masked_pred = pred * combined_mask.cpu().numpy()
            masked_gt_e = gt_e * combined_mask.cpu().numpy()

            pred = torch.from_numpy(pred).unsqueeze(0)
            gt_e = gt_e.unsqueeze(0)
            img_log =  img[0,:,:,:]#.transpose(1,2,0)
            print("img",img.shape)
            if self.test_image_count < 40:
                self.log_images(
                        img_log,
                        masked_pred[0,0,:,:],
                         gt_e[0,:,:],
                        test_dir
                    )
            self.test_image_count += 1

            rmse, mae, std_dev = calculate_metrics(masked_pred.ravel(), masked_gt_e.numpy().ravel())

            self.log(f'test_rmse_step_image_{self.test_image_count}', (rmse * -self.norm_param_depth), on_step=True)
            self.log(f'test_mae_step_image_{self.test_image_count}', (mae * -self.norm_param_depth), on_step=True)
            self.log(f'test_std_dev_step_image_{self.test_image_count}', (std_dev * -self.norm_param_depth), on_step=True)

            self.log('test_rmse_epoch', (rmse * -self.norm_param_depth), on_step=False, on_epoch=True)
            self.log('test_mae_epoch', (mae * -self.norm_param_depth), on_step=False, on_epoch=True)
            self.log('test_std_dev_epoch', (std_dev * -self.norm_param_depth), on_step=False, on_epoch=True)

            self.test_rmse_list.append(rmse * -self.norm_param_depth)
            self.test_mae_list.append(mae * -self.norm_param_depth)
            self.test_std_dev_list.append(std_dev * -self.norm_param_depth)

            print(f'Mean RMSE for image {self.test_image_count} :', rmse * -self.norm_param_depth)
            print(f'Mean MAE for image {self.test_image_count} :', mae * -self.norm_param_depth)
            print(f'Mean Std Dev for image {self.test_image_count} :', std_dev * -self.norm_param_depth)

    def on_train_start(self):
        self.log_results()

    def on_train_epoch_start(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        print(f"Starting epoch {self.current_epoch} - Current learning rate: {current_lr}")

        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []

    def on_validation_epoch_start(self):
        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []

    def on_test_start(self):
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []
        self.test_image_count = 0

    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss / self.train_batch_count
        self.log('train_loss_epoch', avg_train_loss)
        self.total_train_loss = 0.0
        self.train_batch_count = 0
        print(f"Train Loss (Epoch {self.current_epoch}): {avg_train_loss}")

        avg_rmse = torch.tensor(self.epoch_rmse_list).mean() if self.epoch_rmse_list else torch.tensor(0.0)
        avg_mae = torch.tensor(self.epoch_mae_list).mean() if self.epoch_mae_list else torch.tensor(0.0)
        avg_std_dev = torch.tensor(self.epoch_std_dev_list).mean() if self.epoch_std_dev_list else torch.tensor(0.0)

        self.log('avg_train_rmse', avg_rmse)
        self.log('avg_train_mae', avg_mae)
        self.log('avg_train_std_dev', avg_std_dev)

        self.epoch_rmse_list = []
        self.epoch_mae_list = []
        self.epoch_std_dev_list = []

    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_batch_count
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
        self.total_val_loss = 0.0
        self.val_batch_count = 0
        print(f"Validation Loss (Epoch {self.current_epoch}): {avg_val_loss}")

        avg_rmse = torch.tensor(self.val_rmse_list).mean() if self.val_rmse_list else torch.tensor(0.0)
        avg_mae = torch.tensor(self.val_mae_list).mean() if self.val_mae_list else torch.tensor(0.0)
        avg_std_dev = torch.tensor(self.val_std_dev_list).mean() if self.val_std_dev_list else torch.tensor(0.0)

        self.log('avg_val_rmse', avg_rmse)
        self.log('avg_val_mae', avg_mae)
        self.log('avg_val_std_dev', avg_std_dev)

        self.val_rmse_list = []
        self.val_mae_list = []
        self.val_std_dev_list = []

    def on_test_epoch_end(self):
        avg_rmse = torch.tensor(self.test_rmse_list).mean() if self.test_rmse_list else torch.tensor(0.0)
        avg_mae = torch.tensor(self.test_mae_list).mean() if self.test_mae_list else torch.tensor(0.0)
        avg_std_dev = torch.tensor(self.test_std_dev_list).mean() if self.test_std_dev_list else torch.tensor(0.0)

        self.log('avg_test_rmse', avg_rmse)
        self.log('avg_test_mae', avg_mae)
        self.log('avg_test_std_dev', avg_std_dev)

        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_std_dev_list = []

    def on_train_end(self):
        self.writer.close()

    def log_images(self, data: torch.Tensor, predicted_depth: np.ndarray, depth: np.ndarray, dir: str) -> None:
        self.log_results()

        print("data",data.shape)
        print("predicted_depth",predicted_depth.shape)
        print("depth",depth.shape)

        if data.ndim == 3 and data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))
        elif data.ndim == 4 and data.shape[1] == 3:
            data = np.transpose(data[0], (1, 2, 0))
            
        data = data[:, :, [2, 1, 0]] 

        data = np.clip(data, 0, 1)

        predicted_depth = predicted_depth.squeeze()
        gt_depth = depth.squeeze()

        predicted_depth = predicted_depth * -self.norm_param_depth
        gt_depth = gt_depth * -self.norm_param_depth

        combined_min = min(predicted_depth.min(), gt_depth.min())
        combined_max = max(predicted_depth.max(), gt_depth.max())

        if combined_max - combined_min < 1e-6:
            combined_max = combined_min + 1.0

        display_vmin = combined_min
        display_vmax = combined_max

        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(data)
        plt.title("Original RGB")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(gt_depth, cmap='viridis_r', vmin=display_vmin, vmax=display_vmax)
        plt.title("Ground Truth Depth")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        print("predicted_depth_display",predicted_depth.shape)
        plt.subplot(133)
        plt.imshow(predicted_depth, cmap='viridis_r', vmin=display_vmin, vmax=display_vmax)
        plt.title(f"Predicted Depth for model {self.model_type} for region {self.location}")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        abs_save_dir = os.path.abspath(os.path.join(self.run_dir, dir))
        os.makedirs(abs_save_dir, exist_ok=True)

        filename = ""
        if dir == "test_results":
            filename = os.path.join(abs_save_dir, f"depth_comparison_epoch_{self.current_epoch}_image_{self.test_image_count}.png")
        else:
            filename = os.path.join(abs_save_dir, f"depth_comparison_epoch_{self.current_epoch}_batch_{self.global_step}.png")

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def log_results(self):
        if self.run_dir is None:
            run_index = 0
            while os.path.exists(os.path.join(self.base_dir, f"run_{run_index}")):
                run_index += 1
            self.run_dir = os.path.join(self.base_dir, f"run_{run_index}")
            os.makedirs(self.run_dir, exist_ok=True)

    def configure_optimizers(self):
        params_dict = dict(self.projection_head.named_parameters())
        params = []
        base_lr = self.base_lr

        for key, value in params_dict.items():
            if '_D' in key:
                params.append({'params':[value],'lr': base_lr})
            else:
                params.append({'params':[value],'lr': base_lr})

        optimizer = optim.Adam(params, lr=self.base_lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

        return [optimizer], [scheduler]