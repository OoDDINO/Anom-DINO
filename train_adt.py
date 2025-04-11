import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

# Local imports
from config.config import config
from dataset.training.cityscapes import Cityscapes
from dataset.validation.fishyscapes import Fishyscapes
from dataset.validation.lost_and_found import LostAndFound
from dataset.validation.road_anomaly import RoadAnomaly
from dataset.validation.anomaly_score import AnomalyDataset
from dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan
from engine.engine import Engine
from engine.evaluator import SlidingEval
from model.network import Network
from utils.img_utils import Compose, Normalize, ToTensor
from utils.wandb_upload import Tensorboard
from utils.pyt_utils import eval_ood_measure
from utils.logger import *
from adt_net import ADTNet
from losses import ADTNetLoss

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.engine = None
        self.evaluator = None
        self.transform = None
        self.logger = None

    def setup_model(self, ckpt_path):
        """Initialize and load the anomaly detection model"""
        self.model = Network(config.num_classes)
        state_dict = torch.load(ckpt_path)
        state_dict = self._process_state_dict(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        return self.model

    def _process_state_dict(self, state_dict):
        """Process the state dictionary to handle different key formats"""
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        return new_state_dict

    def setup_engine(self, args, config):
        """Setup the training engine"""
        self.logger = logging.getLogger("ours")
        self.logger.propagate = False
        self.engine = Engine(
            custom_arg=args,
            logger=self.logger,
            continue_state_object=config.pretrained_weight_path
        )
        
        self.transform = Compose([
            ToTensor(),
            Normalize(config.image_mean, config.image_std)
        ])
        
        self.evaluator = SlidingEval(
            config,
            device=0 if self.engine.local_rank < 0 else self.engine.local_rank
        )

class ADTNetTrainer:
    def __init__(self, in_channels=1, learning_rate=1e-5, num_epochs=100):
        self.model = ADTNet(in_channels=in_channels).cuda()
        self.criterion = ADTNetLoss(lambda1=1.0, lambda2=1.0, gamma=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.num_epochs = num_epochs
        
    def train(self, train_loader, test_loader):
        best_fpr = float('inf')
        best_prc_auc = 0.0
        
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(train_loader)
            metrics = self._validate(test_loader)
            self._save_best_model(metrics, epoch)
            self.scheduler.step()
            
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            loss = self._process_batch(batch)
            total_loss += loss.item()
            
        return total_loss
    
    def _process_batch(self, batch):
        anomaly_scores = batch['anomaly_score'].cuda()
        labels = batch['ood_gt'].cuda()
        bboxes = batch['bbox']
        
        bbox_masks = self._create_bbox_masks(bboxes, anomaly_scores.shape)
        
        final_prob, T_fg, T_bg, norm_scores = self.model(
            anomaly_scores.unsqueeze(1),
            bbox_masks
        )
        
        loss = self.criterion(
            final_prob * bbox_masks,
            final_prob * (1 - bbox_masks),
            T_fg,
            T_bg,
            labels.unsqueeze(1)
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return loss

    def _validate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            # Validation logic here
            pass
        return {"roc_auc": 0, "prc_auc": 0, "fpr": 0}  # Replace with actual metrics

    def _save_best_model(self, metrics, epoch):
        # Model saving logic here
        pass

def main(gpu, ngpus_per_node, config, args):
    # Initialize detector
    detector = AnomalyDetector()
    detector.setup_engine(args, config)
    model = detector.setup_model(config.rpl_corocl_weight_path)
    
    # Set up distributed training if needed
    if detector.engine.distributed:
        setup_distributed_training(model, detector.engine)
    else:
        model.cuda()
    
    # Generate training data
    train_data = generate_training_data(
        model=model,
        engine=detector.engine,
        test_set=RoadAnomaly(root=config.road_anomaly_root_path, transform=detector.transform),
        data_name='road_anomaly',
        logger=detector.logger
    )
    
    # Initialize and start training
    trainer = ADTNetTrainer()
    trainer.train(*prepare_data_loaders(train_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    add_arguments(parser)
    args = parser.parse_args()
    
    args.world_size = args.nodes * args.gpus
    args.ddp = True if args.world_size > 1 else False
    
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(
            main,
            nprocs=args.gpus,
            args=(args.gpus, config, args)
        )