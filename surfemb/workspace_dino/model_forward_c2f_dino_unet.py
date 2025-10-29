#unet + n objs + larger encoder

import argparse
from typing import Sequence, Union

import torch
from torch.nn import functional as F
import numpy as np
import wandb
import albumentations as A
import cv2
import pytorch_lightning as pl

import torch.nn as nn
import open3d as o3d

from .endecoder import UNet, UNet_Encoder, UNet_Decoder, UNet_Decoder_highdim, UNet_Decoder_highdim_mask, UNet_Decoder_highdim_matching, UNet_Decoder_highdim_matching_512, UNet_Decoder_highdim_matching_256,RealUNet_Decoder_highdim_matching_256

from .. import data
from ..data.obj import Obj

import surfemb.workspace_dino.modules.imagenet as imagenet
import surfemb.workspace_dino.modules.pointnet2 as pointnet2
import surfemb.workspace_dino.modules.layers_pc as layers_pc
from .cross_attention import CrossAttention, SelfAttention ,SurfAttention

import argparse
from typing import Sequence, Union

import torch
from torch.nn import functional as F
import numpy as np
import wandb
import albumentations as A
import cv2
import pytorch_lightning as pl


from ..dep.siren import Siren
from .. import utils

# could be extended to allow other mlp architectures
mlp_class_dict = dict(
    siren=Siren
)

class Gen_corr(pl.LightningModule):
    def __init__(self, objs, n_objs: int, embed_dim=12, lr_mlp=3e-5,
                 mlp_name='siren', mlp_hidden_features=256, mlp_hidden_layers=2,
                 warmup_steps=100, fusion_token_len=256, image_size=224, patch_size=16,
                 num_points=10000, dino_model=None, one_obj_idx = None,
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.dino_model = dino_model
        
        # CHASIS object paths
        cad_path = "/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models/obj_000001.ply"
        feat_path = "/media/sujith/Project/NOARK_CV/RAG-6DPose-code/cad_features/obj_000001_dino_feat.pt"

        # Load CHASIS point cloud
        pcd = o3d.io.read_point_cloud(cad_path)
        self.p1_ori = torch.tensor(np.asarray(pcd.points)).float()
        self.p1_color = torch.tensor(np.asarray(pcd.colors)).float()

        # Load CHASIS DINO features
        self.pd1_ori = torch.load(feat_path).float()

        # For compatibility with 3-object code, duplicate to p5 and p6
        self.p5_ori = self.p1_ori.clone()
        self.p6_ori = self.p1_ori.clone()
        self.p5_color = self.p1_color.clone()
        self.p6_color = self.p1_color.clone()
        self.pd5_ori = self.pd1_ori.clone()
        self.pd6_ori = self.pd1_ori.clone()


        self.p1 = None
        self.p5 = None
        self.p6 = None
        
        self.pd1 = None
        self.pd5 = None
        self.pd6 = None
        
        self.pc1 = None
        self.pc5 = None
        self.pc6 = None
        
        self.point_clouds_list = [] # (n_objs, m, 3) 归一到m个点
        self.pcd_dino = []  # (n_objs, m, embed_dim)
        self.pcd_color = []

        self.embed_dim = embed_dim
        self.lr_mlp =  lr_mlp
        self.warmup_steps = warmup_steps
        self.n_objs = n_objs
        
        self.node_b_num = 512+768 # 和image feature dim一致
        self.fusion_token_len = self.node_b_num
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_points = num_points
        self.embed_dim = embed_dim

        self.pcd_sizes = [self.p1_ori.size(0),self.p5_ori.size(0),self.p6_ori.size(0)]
        
        # self.p_sz = self.pcd_sizes[one_obj_idx]
        
        np.random.seed(42)
        
        self.selected_points_num = 3000
        
        self.random_indices_1 = np.random.choice(self.pcd_sizes[0], size=self.selected_points_num, replace=False)
        self.random_indices_2 = np.random.choice(self.pcd_sizes[1], size=self.selected_points_num, replace=False)
        self.random_indices_3 = np.random.choice(self.pcd_sizes[2], size=self.selected_points_num, replace=False)

        
        # self.cloud_proj1 = torch.nn.Linear(self.p_sz // 4, self.fusion_token_len, bias=True,dtype=torch.float32)
        
        
        self.scales = [objs[0].scale, objs[1].scale, objs[2].scale]
        self.offsets = [objs[0].offset, objs[1].offset, objs[2].offset]
        
        self.surf_samples = [self.p1_ori[self.random_indices_1], self.p5_ori[self.random_indices_2], self.p6_ori[self.random_indices_3]]

        self.surf_all_points = [self.p1_ori, self.p5_ori, self.p6_ori]
        
        for idx in range(len(self.surf_samples)):
            # print(self.surf_samples[idx].shape) torch.Size([3000, 3])
            # print(self.scales[idx].shape) float 
            self.surf_samples[idx] = (self.surf_samples[idx] - self.offsets[idx]) / self.scales[idx]
        
        for idx in range(len(self.surf_all_points)):
            # print(self.surf_samples[idx].shape) torch.Size([3000, 3])
            # print(self.scales[idx].shape) float 
            self.surf_all_points[idx] = (self.surf_all_points[idx] - self.offsets[idx]) / self.scales[idx]   
        
        self.pcd_cat_1 = torch.cat([(self.p1_ori- self.offsets[0])/self.scales[0], self.p1_color/255, self.pd1_ori],dim=-1)
        self.pcd_cat_1 = self.pcd_cat_1[self.random_indices_1].float()
        self.pcd_cat_2 = torch.cat([(self.p5_ori- self.offsets[1])/self.scales[1], self.p5_color/255, self.pd5_ori],dim=-1)
        self.pcd_cat_2 = self.pcd_cat_2[self.random_indices_2].float()
        self.pcd_cat_3 = torch.cat([(self.p6_ori- self.offsets[2])/self.scales[2], self.p6_color/255, self.pd6_ori],dim=-1)
        self.pcd_cat_3 = self.pcd_cat_3[self.random_indices_3].float()
        
        
        # self.pcd_feat_select_points = [self.pcd_cat_1, self.pcd_cat_2, self.pcd_cat_3, self.pcd_cat_4, self.pcd_cat_5, self.pcd_cat_6, self.pcd_cat_7, self.pcd_cat_8]
        
        self.self_att_1 = SelfAttention(dim=self.selected_points_num, fusion_dim=self.fusion_token_len)

        
        # self.u_encoder = imagenet.convnext()
        self.u_encoder = imagenet.ResEncoder()
        self.u_decoder = RealUNet_Decoder_highdim_matching_256(input_channel = 2560, out_channel=256)
        self.mask_decoder = UNet_Decoder_highdim_mask(input_channel = 2560)
        # self.pc_encoder=pointnet2.PCEncoder(opt,Ca=64,Cb=256,Cg=512)
        self.img_encoder=imagenet.ImageEncoder()

        self.H_fine_res = 28
        self.W_fine_res = 28
        self.node_b_attention_pn = layers_pc.PointNet(512+768+3+3,
                                               [1024, self.H_fine_res*self.W_fine_res],
                                               )

        self.img_32_attention_conv=nn.Sequential(nn.Conv2d(512+768+256,1024,1,bias=False),nn.BatchNorm2d(1024),nn.ReLU(),
                                                    nn.Conv2d(1024,1024,1,bias=False),nn.BatchNorm2d(1024),nn.ReLU(),
                                                    nn.Conv2d(1024,self.node_b_num,1,bias=False))
        
        self.cross_att = CrossAttention(dim=self.node_b_num)
        
        
        self.transposed_conv = nn.ConvTranspose2d(768, 768, kernel_size=3, stride=2, padding=3, output_padding=1)
        
        
        self.emb_dim = 128
        self.n_pos, self.n_neg = 2048, 2048
        # self.lr_cnn, self.lr_mlp = lr_cnn, lr_mlp
        # self.warmup_steps = warmup_steps
        self.key_noise = 1e-3
        self.separate_decoders = True

        # query model
        # self.cnn = ResNetUNet(
        
        # key models
        mlp_class = mlp_class_dict[mlp_name]
        mlp_args = dict(in_features=3, out_features=self.emb_dim,
                        hidden_features=mlp_hidden_features, hidden_layers=mlp_hidden_layers)
        self.mlps = torch.nn.Sequential(*[mlp_class(**mlp_args) for _ in range(n_objs)])

       
        mlp_gt_corr_args = dict(in_features=768, out_features=256,
                        hidden_features=384, hidden_layers=mlp_hidden_layers)
        self.mlps_gt_corr = torch.nn.Sequential(*[mlp_class(**mlp_gt_corr_args) for _ in range(n_objs)])
        

        self.dino_parm_blocks_1 = nn.Parameter(self.pcd_cat_1.float().clone(), requires_grad=True)
        self.dino_parm_blocks_5 = nn.Parameter(self.pcd_cat_2.float().clone(), requires_grad=True)
        self.dino_parm_blocks_6 = nn.Parameter(self.pcd_cat_3.float().clone(), requires_grad=True)

        
        
        self.dino_parm_blocks_LIST = [self.dino_parm_blocks_1, self.dino_parm_blocks_5, self.dino_parm_blocks_6]
        
        
        mlp_fusion_args = dict(in_features=384, out_features=256,
                        hidden_features=256, hidden_layers=mlp_hidden_layers)
        self.mlps_fusion = torch.nn.Sequential(*[mlp_class(**mlp_fusion_args) for _ in range(n_objs)])
        
        
        
    def configure_optimizers(self):
        param_groups = []
        param_groups.append({"params": [p for p in self.parameters() if p.requires_grad],
                         'lr': 3e-5, 'weight_decay': 0.0}  )

        opt = torch.optim.Adam(
            param_groups
        )
        sched = dict(
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / self.warmup_steps)),
            interval='step'
        )
        # https://github.com/clovaai/donut/issues/255
        return [opt], [sched]
    
    def get_auxs(self, objs: Sequence[Obj], crop_res: int):
        
        random_crop_aux = data.std_auxs.RandomRotatedMaskCrop(crop_res, max_angle=0,
                offset_scale=0 if True else 1,
                use_bbox=False,
                rgb_interpolation=(cv2.INTER_LINEAR,),
                )
        return (
            data.std_auxs.RgbLoader(),
            data.std_auxs.MaskLoader(),
            random_crop_aux.definition_aux,
            # Some image augmentations probably make most sense in the original image, before rotation / rescaling
            # by cropping. 'definition_aux' registers 'AABB_crop' such that the "expensive" image augmentation is only
            # performed where the crop is going to be taken from.
            data.std_auxs.TransformsAux(key='rgb', tfms=A.Compose([
                A.GaussianBlur(blur_limit=(1, 3)),
                A.ISONoise(),
                A.GaussNoise(),
                data.tfms.DebayerArtefacts(),
                data.tfms.Unsharpen(),
                A.CLAHE(),  # could probably be moved to the post-crop augmentations
                A.GaussianBlur(blur_limit=(1, 3)),
            ])),
            random_crop_aux.apply_aux,
            data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=False),
            data.pose_auxs.SurfaceSampleAux(objs, self.n_neg),
            data.pose_auxs.MaskSamplesAux(self.n_pos),
            data.std_auxs.TransformsAux(tfms=A.Compose([
                A.CoarseDropout(max_height=16, max_width=16, min_width=8, min_height=8),
                A.ColorJitter(hue=0.1),
            ])),
            data.std_auxs.NormalizeAux(),
            data.std_auxs.KeyFilterAux({'rgb_crop', 'obj_coord', 'obj_idx', 'K_crop', 'img_id','scene_id','mask_visib_crop','K','bbox_visib','bbox','M_crop','mask_samples','surface_samples'})
        )
        
        
    def get_infer_auxs(self, objs: Sequence[Obj], crop_res: int, from_detections=True):
        auxs = [data.std_auxs.RgbLoader()]
        if not from_detections:
            auxs.append(data.std_auxs.MaskLoader())
        auxs.append(data.std_auxs.RandomRotatedMaskCrop(
            crop_res, max_angle=0,
            offset_scale=0 if from_detections else 1,
            use_bbox=from_detections,
            rgb_interpolation=(cv2.INTER_LINEAR,),
        ))
        if not from_detections:
            auxs += [
                data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=True),
                data.pose_auxs.SurfaceSampleAux(objs, self.n_neg),
                data.pose_auxs.MaskSamplesAux(self.n_pos),
            ]
        auxs.append(data.std_auxs.NormalizeAux())
        return auxs
    
    def get_auxs_infer(self, objs: Sequence[Obj], crop_res: int):
        return (
            data.std_auxs.RgbLoader(),
            data.std_auxs.MaskLoader(),
            
            data.std_auxs.RandomRotatedMaskCrop(crop_res, max_angle=0,
                offset_scale=0 if True else 1,
                use_bbox=False,
                rgb_interpolation=(cv2.INTER_LINEAR,),), ####
            data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=False), ####################not replace is visib mask
            data.std_auxs.NormalizeAux(),
            data.std_auxs.KeyFilterAux({'rgb_crop', 'obj_coord', 'obj_idx','K_crop', 'img_id','scene_id','mask_visib_crop','K','bbox_visib','bbox','M_crop'})
        )
    


    def step(self, batch, log_prefix):
        
        img = batch['rgb_crop'].float().permute(0,3,1,2)  # (B, 3, H, W)
        gt_coord_img = batch['obj_coord']  # (B, H, W, 4) [-1, 1] dim=3 is gtmask

        device = img.device
        B, _, H, W = img.shape
        img_dino_feat = self.dino_model(img)
        # img_dino_feat = torch.nn.functional.interpolate(img_dino_feat, size=(28, 28), mode='bilinear', align_corners=False)
        img_dino_feat = self.transposed_conv(img_dino_feat)
        # print(img_dino_feat.shape)
        

        q_1 = self.pcd_cat_1.T.to(device)
        k_1 = self.pcd_cat_1.T.to(device)
        v_1 = self.pcd_cat_1.T.to(device)
        self.p1 = self.self_att_1(q_1, k_1, v_1)
        
        q_2 = self.pcd_cat_2.T.to(device)
        k_2 = self.pcd_cat_2.T.to(device)
        v_2 = self.pcd_cat_2.T.to(device)
        self.p2 = self.self_att_1(q_2, k_2, v_2)
        
        q_3 = self.pcd_cat_3.T.to(device)
        k_3 = self.pcd_cat_3.T.to(device)
        v_3 = self.pcd_cat_3.T.to(device)
        self.p3 = self.self_att_1(q_3, k_3, v_3)
        
        self.point_clouds_list = [self.p1,self.p2,self.p3]
        
        
        batch_pcd_xyz = torch.zeros((B, self.point_clouds_list[0].size(0), self.p1.size(-1))).to(device)
        batch_select_xyz = torch.zeros((B, self.surf_samples[0].size(0), 3)).to(device)
        coords_neg = torch.zeros((B, self.surf_samples[0].size(0), 3)).to(device)

        batch_all_feat = torch.zeros((B, self.dino_parm_blocks_LIST[batch['obj_idx'][0]].size(0), self.dino_parm_blocks_LIST[batch['obj_idx'][0]].size(1))).requires_grad_(True).to(device)

        for i in range(B):
            idx = batch['obj_idx'][i]
            batch_pcd_xyz[i] = self.point_clouds_list[idx]
            coords_neg[i] = self.surf_samples[idx]
            batch_select_xyz[i] = self.surf_samples[idx]
            
            batch_all_feat[i] = self.dino_parm_blocks_LIST[idx].requires_grad_(True)
            
        batch_all_feat[i] = batch_all_feat[i].requires_grad_(True)
        
      
        # RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[16, 224, 224, 3] to have 3 channels, but got 224 channels instead
        img_feature_set=self.img_encoder(img)
        img_global_feature=img_feature_set[-1]  #512

        
        img_res_feat_list = self.u_encoder(img)
        img_s32_feature_map = img_res_feat_list[2]
        # img_s32_feature_map = img_feature_set[-4]
        
        img_s32_feature_map = torch.cat([img_s32_feature_map, img_dino_feat],dim=1) # 512+768

        pcd_feat = batch_pcd_xyz
        # print(pcd_feat.shape) # torch.Size([16, 256, 13])
        
        ## a. pcf + image global / mlp
        Mb = self.node_b_num #512
        C_img=img_global_feature.size(1)
        # img_s32_feature_map_BCHw=img_s32_feature_map.view(B,img_s32_feature_map.size(1),-1)
        img_global_feature_BCMb = img_global_feature.squeeze(3).expand(B, C_img, Mb)  # BxC_img -> BxC_imgxMb

        node_b_features = pcd_feat
        
        # print(node_b_features.shape)
        # print(img_global_feature_BCMb.shape)
        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb
        # print(node_b_attention_score.shape)
        query = img_s32_feature_map.view(B, 28*28, -1)
        key = node_b_attention_score
        value = node_b_attention_score
        img_32_attention = self.cross_att(query, key, value)

        mul_att_feat = img_32_attention.view(B,-1, 28,28)

        fusion_feature_todecode = torch.cat([mul_att_feat, img_s32_feature_map],dim=1) # 256+512+768

        
        decoded = self.u_decoder(fusion_feature_todecode,img_res_feat_list)
        pred_mask = self.mask_decoder(fusion_feature_todecode)

        gt_mask = batch['obj_coord'][:,:,:,3:4]
        gt_mask = gt_mask.permute(0,3,1,2)
        
        
        pred_mask = F.sigmoid(pred_mask)

        mask_loss_Func = nn.L1Loss()
        mask_loss = mask_loss_Func(pred_mask, gt_mask)

        mask_samples = batch['mask_samples']  # (B, n_pos, 2)
        obj_idx = batch['obj_idx']  # (B,)
        # coords_neg =   # (B, n_neg, 3) [-1, 1]
        coords_neg = batch['surface_samples']
        
        assert coords_neg.shape[1] == self.n_neg
        y, x = mask_samples.permute(2, 0, 1)  # 2 x (B, n_pos)

        def find_nearest_points(t1, t2):
            dists = torch.cdist(t1, t2)
            nearest_indices = dists.view(B, self.n_neg, -1).argmin(dim=-1)

            return nearest_indices.to(device)

        queries = decoded[torch.arange(B).view(B, 1), :, y, x]  # (B, n_pos, emb_dim)

        # compute similarities for positive pairs
        coords_pos = gt_coord_img[torch.arange(B).view(B, 1), y, x, :3]  # (B, n_pos, 3) [-1, 1]
        coords_pos += torch.randn_like(coords_pos) * self.key_noise
        
        
        xyz_gt_coord = coords_pos
        idx_gt_coord = find_nearest_points(xyz_gt_coord, batch_select_xyz).to(device) ##########change to sel
        # dino_gt_coord (B, n_neg, feat_dim1)
        dino_gt_coord = torch.gather(input=batch_all_feat, dim=1, index=idx_gt_coord.unsqueeze(-1).expand(B, self.n_neg, 768)).requires_grad_(True).to(device)

        dino_key_gt_coord = torch.stack([self.mlps_gt_corr[i](c) for i, c in zip(obj_idx, dino_gt_coord)])
        # print(dino_key_gt_coord.std(),dino_key_gt_coord.mean())
        keys_pos = torch.stack([self.mlps_fusion[i](torch.cat([self.mlps[i](c), d], dim=-1)) for i, c, d in zip(obj_idx, coords_pos, dino_key_gt_coord)])  # (B, n_pos, emb_dim)
        # print(keys_pos.std(),keys_pos.mean())
        sim_pos = (queries * keys_pos).sum(dim=-1, keepdim=True)  # (B, n_pos, 1)
        
        
        # compute similarities for negative pairs
        coords_neg += torch.randn_like(coords_neg) * self.key_noise
        
        xyz_surface_samples = coords_neg
        idx_surface_samples = find_nearest_points(xyz_surface_samples, batch_select_xyz).to(device) ##########change to sel
        # dino_surface_samples (B, n_neg, feat_dim1)
        dino_surface_samples = torch.gather(input=batch_all_feat, dim=1, index=idx_surface_samples.unsqueeze(-1).expand(B, self.n_neg, 768)).requires_grad_(True).to(device)
        # dino_surface_samples_att = self.att_gt_corr(dino_surface_samples,dino_surface_samples,dino_surface_samples)

        dino_key_surface_samples = torch.stack([self.mlps_gt_corr[i](c) for i, c in zip(obj_idx, dino_surface_samples)])
        # print(dino_key_surface_samples.std(),dino_key_surface_samples.mean())
        keys_neg = torch.stack([self.mlps_fusion[i](torch.cat([self.mlps[i](v), d],dim=-1)) for i, v, d in zip(obj_idx, coords_neg, dino_key_surface_samples)])  # (B, n_neg, n_dim)
        # print(keys_neg.std(),keys_neg.mean())
        sim_neg = queries @ keys_neg.permute(0, 2, 1)  # (B, n_pos, n_neg)

        lgts = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)  # (B, 1 + n_neg, n_pos)
        target = torch.zeros(B, self.n_pos, device=device, dtype=torch.long)
        nce_loss = F.cross_entropy(lgts, target)
        # print(nce_loss)
        loss = mask_loss + nce_loss
        

        
        self.log(f'{log_prefix}/loss', loss)
        self.log(f'{log_prefix}/nce_loss', nce_loss)
        self.log(f'{log_prefix}/mask_loss', mask_loss)

        
        return loss
        

    @torch.no_grad()
    def infer_cnn(self, img: Union[np.ndarray, torch.Tensor], obj_idx, rotation_ensemble=True):
        assert not self.training
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = data.tfms.normalize(img)
            img = torch.from_numpy(img).to(self.device)
        # _, h, w = img.shape
        if rotation_ensemble:
            img = utils.rotate_batch(img.permute(2,0,1))  # (4, 3, h, h)
        else:
            img = img[None]  # (1, 3, h, w)
        img = img.float()
        device = self.device
        B, _, H, W = img.shape
        img_dino_feat = self.dino_model(img)
        img_dino_feat = self.transposed_conv(img_dino_feat)
        
        
        q_1 = self.pcd_cat_1.T.to(device)
        k_1 = self.pcd_cat_1.T.to(device)
        v_1 = self.pcd_cat_1.T.to(device)
        self.p1 = self.self_att_1(q_1, k_1, v_1)
        
        q_2 = self.pcd_cat_2.T.to(device)
        k_2 = self.pcd_cat_2.T.to(device)
        v_2 = self.pcd_cat_2.T.to(device)
        self.p2 = self.self_att_1(q_2, k_2, v_2)
        
        q_3 = self.pcd_cat_3.T.to(device)
        k_3 = self.pcd_cat_3.T.to(device)
        v_3 = self.pcd_cat_3.T.to(device)
        self.p3 = self.self_att_1(q_3, k_3, v_3)
        
        
        self.point_clouds_list = [self.p1,self.p2,self.p3]
        
        
        batch_pcd_xyz = torch.zeros((B, self.point_clouds_list[0].size(0), self.p1.size(-1))).to(device)

        
        for i in range(B):
            idx = obj_idx
            batch_pcd_xyz[i] = self.point_clouds_list[idx]

        
        # RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[16, 224, 224, 3] to have 3 channels, but got 224 channels instead
        img_feature_set=self.img_encoder(img)
        img_global_feature=img_feature_set[-1]  #512

        
        img_res_feat_list = self.u_encoder(img)
        img_s32_feature_map = img_res_feat_list[2]
        # img_s32_feature_map = img_feature_set[-4]
        
        img_s32_feature_map = torch.cat([img_s32_feature_map, img_dino_feat],dim=1) # 512+768
        
        pcd_feat = batch_pcd_xyz
        # print(pcd_feat.shape) # torch.Size([16, 256, 13])
        
        ## a. pcf + image global / mlp
        Mb = self.node_b_num #512
        C_img=img_global_feature.size(1)
        # img_s32_feature_map_BCHw=img_s32_feature_map.view(B,img_s32_feature_map.size(1),-1)
        img_global_feature_BCMb = img_global_feature.squeeze(3).expand(B, C_img, Mb)  # BxC_img -> BxC_imgxMb

        node_b_features = pcd_feat
        
        # print(node_b_features.shape)
        # print(img_global_feature_BCMb.shape)
        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb
        # print(node_b_attention_score.shape)
        query = img_s32_feature_map.view(B, 28*28, -1)
        key = node_b_attention_score
        value = node_b_attention_score
        img_32_attention = self.cross_att(query, key, value)

        mul_att_feat = img_32_attention.view(B,-1, 28,28)
        # img_s32_feature_map_fusion=torch.cat((torch.sum(img_32_attention.unsqueeze(1)*node_b_features.unsqueeze(-1).unsqueeze(-1),dim=2),img_s32_feature_map),dim=1)    #(B,512+256,H,W)
        # print(mul_att_feat.shape)
        # print(img_s32_feature_map.shape)    
        fusion_feature_todecode = torch.cat([mul_att_feat, img_s32_feature_map],dim=1) # 256+512+768

        
        decoded = self.u_decoder(fusion_feature_todecode,img_res_feat_list)
        pred_mask = self.mask_decoder(fusion_feature_todecode)


        cnn_out = decoded
        
        if rotation_ensemble:
            cnn_out = utils.rotate_batch_back(cnn_out).mean(dim=0)
            pred_mask = utils.rotate_batch_back(pred_mask).mean(dim=0)
        else:
            cnn_out = cnn_out[0]
            pred_mask = pred_mask[0]
        mask_lgts, query_img = pred_mask[0], cnn_out
        query_img = query_img.permute(1, 2, 0)  # (h, w, emb_dim)
        return mask_lgts, query_img

    @torch.no_grad()
    def infer_mlp_3x64(self, pts_norm: Union[np.ndarray, torch.Tensor], obj_idx):
        assert not self.training
        if isinstance(pts_norm, np.ndarray):
            pts_norm = torch.from_numpy(pts_norm).to(self.device).float()
        return self.mlps[obj_idx](pts_norm)  # (..., emb_dim)
    
    
    @torch.no_grad()
    def infer_mlp(self, pts_norm: Union[np.ndarray, torch.Tensor], obj_idx):
        if isinstance(pts_norm, np.ndarray):
            pts_norm = torch.from_numpy(pts_norm).to(self.device).float()
            
        # coords_neg =   # (n_neg, 3) [-1, 1]
        coords_neg = pts_norm
        device = self.device
        
        def find_nearest_points(t1, t2):
            dists = torch.cdist(t1, t2)
            nearest_indices = dists.view(pts_norm.size(0), -1).argmin(dim=-1)

            return nearest_indices.to(device)
        
        xyz_surface_samples = coords_neg
        
        # batch_all_xyz = self.surf_all_points[obj_idx].to(device).float()
        batch_select_xyz = self.surf_samples[obj_idx].to(device).float()
        
        idx_surface_samples = find_nearest_points(xyz_surface_samples, batch_select_xyz).to(device)
        # dino_surface_samples (B, n_neg, feat_dim1)
        batch_all_feat = self.dino_parm_blocks_LIST[obj_idx]
        dino_surface_samples = torch.gather(input=batch_all_feat.to(device), dim=0, index=idx_surface_samples.unsqueeze(-1).expand(pts_norm.size(0), 768)).to(device)
        dino_surface_samples = dino_surface_samples.unsqueeze(0)

        dino_key_surface_samples = self.mlps_gt_corr[obj_idx](dino_surface_samples).to(device)
        # print(dino_key_surface_samples.std(),dino_key_surface_samples.mean())
        keys_neg = self.mlps_fusion[obj_idx](torch.cat([self.mlps[obj_idx](coords_neg).unsqueeze(0), dino_key_surface_samples],dim=-1))   # (B, n_neg, n_dim)    
        

        return keys_neg.squeeze(0).to(device)
    
    

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        # self.log_image_sample(batch)
        return self.step(batch, 'valid')

    
    

