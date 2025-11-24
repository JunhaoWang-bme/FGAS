import os
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model import get_model
from data_loader import get_multi_view_data_loaders, get_data_loaders
from losses import get_loss_function, get_contrastive_loss_function


class ContrastiveTrainer:
    
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # create model
        self.model = get_model(config).to(device)
        
        # loss function
        self.segmentation_loss = get_loss_function(config)
        self.contrastive_loss = get_contrastive_loss_function(config)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        if self.config['training']['scheduler']['type'] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config['training']['scheduler']['patience'],
                factor=self.config['training']['scheduler']['factor'],
                min_lr=self.config['training']['scheduler']['min_lr']
            )
        elif self.config['training']['scheduler']['type'] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['scheduler']['T_max'],
                eta_min=self.config['training']['scheduler']['eta_min']
            )
        else:
            self.scheduler = None

        self.writer = SummaryWriter(config['paths']['log_dir'])
        self.contrastive_weight = config['training']['contrastive_weight']
        self.segmentation_weight = config['training']['segmentation_weight']

    
    def _create_scheduler(self):
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr']
            )
        elif scheduler_type == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_contrastive_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            sag_images = batch['sag'].to(self.device)
            cor_images = batch['cor'].to(self.device)
            tra_images = batch['tra'].to(self.device)
            labels = batch['label'].to(self.device)
            patient_ids = batch['patient_id']

            self.optimizer.zero_grad()

            (sag_logits, cor_logits, tra_logits), (sag_features, cor_features, tra_features) = \
                self.model.forward_multi_view(sag_images, cor_images, tra_images)

            seg_loss = self.segmentation_loss(sag_logits, labels)

            patient_id_tensor = torch.tensor([int(pid) for pid in patient_ids], device=self.device)
            contrastive_loss, intra_loss, inter_loss = self.contrastive_loss(
                sag_features, cor_features, tra_features, patient_id_tensor
            )

            # loss
            total_loss_batch = (
                self.segmentation_weight * seg_loss + 
                self.contrastive_weight * contrastive_loss
            )

            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_seg_loss += seg_loss.item()
            total_contrastive_loss += contrastive_loss.item()

            progress_bar.set_postfix({
                'Total Loss': f'{total_loss_batch.item():.4f}',
                'Seg Loss': f'{seg_loss.item():.4f}',
                'Contrastive Loss': f'{contrastive_loss.item():.4f}'
            })

            if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_total_loss = total_loss / len(train_loader)
        avg_seg_loss = total_seg_loss / len(train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(train_loader)
        
        return avg_total_loss, avg_seg_loss, avg_contrastive_loss
    
    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_contrastive_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                sag_images = batch['sag'].to(self.device)
                cor_images = batch['cor'].to(self.device)
                tra_images = batch['tra'].to(self.device)
                labels = batch['label'].to(self.device)
                patient_ids = batch['patient_id']

                (sag_logits, cor_logits, tra_logits), (sag_features, cor_features, tra_features) = \
                    self.model.forward_multi_view(sag_images, cor_images, tra_images)

                seg_loss = self.segmentation_loss(sag_logits, labels)

                patient_id_tensor = torch.tensor([int(pid) for pid in patient_ids], device=self.device)
                contrastive_loss, _, _ = self.contrastive_loss(
                    sag_features, cor_features, tra_features, patient_id_tensor
                )
                
                # loss
                total_loss_batch = (
                    self.segmentation_weight * seg_loss + 
                    self.contrastive_weight * contrastive_loss
                )

                total_loss += total_loss_batch.item()
                total_seg_loss += seg_loss.item()
                total_contrastive_loss += contrastive_loss.item()

        avg_total_loss = total_loss / len(val_loader)
        avg_seg_loss = total_seg_loss / len(val_loader)
        avg_contrastive_loss = total_contrastive_loss / len(val_loader)
        
        return avg_total_loss, avg_seg_loss, avg_contrastive_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_total_loss, train_seg_loss, train_contrastive_loss = self.train_epoch(train_loader, epoch)

            val_total_loss, val_seg_loss, val_contrastive_loss = self.validate_epoch(val_loader, epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_total_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # TensorBoard
            self.writer.add_scalar('Loss/Train_Total', train_total_loss, epoch)
            self.writer.add_scalar('Loss/Train_Segmentation', train_seg_loss, epoch)
            self.writer.add_scalar('Loss/Train_Contrastive', train_contrastive_loss, epoch)
            self.writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
            self.writer.add_scalar('Loss/Val_Segmentation', val_seg_loss, epoch)
            self.writer.add_scalar('Loss/Val_Contrastive', val_contrastive_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # save model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                self.save_checkpoint(epoch, val_total_loss, is_best=True)

            # checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_total_loss, is_best=False)
        
        self.writer.close()
        print("\n OK")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_model_path = os.path.join(checkpoint_dir, self.config['paths']['best_model_name'])
            torch.save(checkpoint, best_model_path)


class StandardTrainer:
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = get_model(config).to(device)
        self.criterion = get_loss_function(config)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = self._create_scheduler()
        self.writer = SummaryWriter(config['paths']['log_dir'])
    
    def _create_scheduler(self):
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr']
            )
        elif scheduler_type == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate_epoch(val_loader, epoch)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
        
        self.writer.close()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_model_path = os.path.join(checkpoint_dir, self.config['paths']['best_model_name'])
            torch.save(checkpoint, best_model_path)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_contrastive = config['model'].get('use_contrastive', False)
    
    if use_contrastive:
        train_loader, val_loader, label_classes = get_multi_view_data_loaders(config)
        trainer = ContrastiveTrainer(config, device)
        
        # Start Training
        trainer.train(train_loader, val_loader, config['training']['num_epochs'])
    else:
        train_loader, val_loader, label_classes = get_data_loaders(config)
        trainer = StandardTrainer(config, device)
        
        # Start Training
        trainer.train(train_loader, val_loader, config['training']['num_epochs'])


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    main()