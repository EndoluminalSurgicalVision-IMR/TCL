from __future__ import absolute_import, division, print_function

import time
import numpy as np
import torch
import torch.nn.functional as F


class Trainer:
    """
    We utilize the code architecture of AF-SfMlearner/Monodepth2.
    You can embed the following module into your code based on the specific SfM baseline structure and dataset.
    Please add additional basic functions according to your specific requirements.
    We present the code directly related to the AiC module.
    """
    def __init__(self, options):
        """
        Please follow AF-SfMlearner/Monodepth2 structure to input the options
            and define them in the options.py file.
        Define your own network architecture, dataloader, and model parameters:
        """
        self.opt = options

    def train(self):
        """Run the entire training pipeline

        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.val = 0
        for self.epoch in range(self.opt.num_epochs):
            ## !!! Note :Please validate the entire validation set after each epoch, rather than validating after each batch.
            self.train_epoch()
            self.val_epoch()

        ### other process(log,save model......)


    def train_epoch(self):
        """Run a single epoch of training and validation
        """
        self.w_lda = self.opt.depth_wl_weight
        self.w_lpa = self.opt.pose_wl_weight
        for batch_idx, inputs in enumerate(self.train_loader):
            self.seed = 10010 + self.step
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            self.set_train()
            outputs, losses = self.process_batch(inputs)

            ####
            # Other functions(optimize,step,log...)
            ####


    def val_epoch(self):
        def val_epoch(self):
            """Run a single epoch of validation
               !!! Note :Please validate the entire validation set after each epoch, rather than validating after each batch.
            """

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses

        """
        outputs ={}
        ####
        # Other basic functions

        # In endoscopic scenarios, the endoscope may move towards any direction.
        # Please utilize the pose estimation function from AF-SfMlearner: self.predict_poses_AF.
        ####


        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def compute_losses(self, inputs, outputs):
        """
        Here shows how the AiC module is applied to photometric loss.
        Please add other loss functions as needed.
        """

        losses = {}
        total_loss = 0

        total_repro_loss = 0
        total_disp_smooth_loss = 0

        for scale in self.opt.scales:

            loss = 0
            loss_reprojection = 0

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # generate triplet mask
            mask_ref_illu_list = self.get_ref_illu_mask(inputs, outputs)
            outputs[("mask_ref_illu", -1)] = mask_ref_illu_list[0].detach()
            outputs[("mask_ref_illu", 1)] = mask_ref_illu_list[1].detach()


            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses = self.compute_reprojection_loss(pred, target) * outputs[
                    ("mask_ref_illu", frame_id)]
                reprojection_losses = reprojection_losses * inputs['valid_mask'] * outputs[
                    ("valid_mask", frame_id, scale)]
            ### Compute the average of the photometric loss using two reference frames.
                loss_reprojection += reprojection_losses.mean()

            loss_repro_scale = (loss_reprojection / 2.0).item()
            total_repro_loss += loss_repro_scale
            loss += loss_reprojection / 2.0
            #### other loss functions



            # loss summary
            total_loss += loss


    def get_ref_illu_mask(self, inputs, outputs):

        list_num = self.opt.illu_listnum

        target = inputs[("color", 0, 0)]
        tgt_feature = self.get_encode_features(target, list_num)

        ref_warp_feature_0 = self.get_encode_features(outputs[("color", -1, 0)], list_num)
        ref_warp_feature_1 = self.get_encode_features(outputs[("color", 1, 0)], list_num)

        avg_illumination_f = self.get_avg_illumination(tgt_feature, ref_warp_feature_0, ref_warp_feature_1)
        mask_ref_illu_list = self.get_mask_ref_illu(tgt_feature, ref_warp_feature_0, ref_warp_feature_1,
                                                    avg_illumination_f)

        return mask_ref_illu_list

    def get_encode_features(self,tensor, list_num=0):
        _, _, height, width = tensor.size()
        feature =self.models["encoder"](tensor)
        # list_num=0 :bx64x128x160
        f_av = feature[list_num].mean(dim=1, keepdim=True)
        features_scaled = F.interpolate(f_av, (height, width), mode="bilinear", align_corners=True)

        return features_scaled


    def get_avg_illumination(self,tgt_feature ,ref_warp_feature_0,ref_warp_feature_1):

        diff_f_0=torch.abs(tgt_feature - ref_warp_feature_0)
        diff_f_1 = torch.abs(tgt_feature - ref_warp_feature_1)
        f_ref_av=(diff_f_0>diff_f_1)* ref_warp_feature_1+(diff_f_0<=diff_f_1)* ref_warp_feature_0
        avg_illumination_f = (tgt_feature +f_ref_av) / 2

        return avg_illumination_f


    def get_mask_ref_illu(self,ref_warp_feature_0, ref_warp_feature_1, avg_illumination_f):
        tm_low = self.opt.tm_thre
        mask_ref_illu_list = []

        ssim_weight=self.opt.illu_ssimweight

        ssim_ref_0 = 1 - self.norm_tensor_each(self.ssim(ref_warp_feature_0, avg_illumination_f))
        ssim_ref_1 = 1 - self.norm_tensor_each(self.ssim(ref_warp_feature_1, avg_illumination_f))

        diff_warped_f_0 = 1 - self.norm_tensor_each((ref_warp_feature_0 - avg_illumination_f).abs().clamp(0, 1))
        diff_warped_f_1 = 1 - self.norm_tensor_each((ref_warp_feature_1 - avg_illumination_f).abs().clamp(0, 1))

        diff_0 = self.norm_tensor_each(ssim_ref_0 * ssim_weight + diff_warped_f_0 * (1 - ssim_weight))
        diff_1 = self.norm_tensor_each(ssim_ref_1 * ssim_weight + diff_warped_f_1 * (1 - ssim_weight))

        norm_ref_mask_0 = (self.norm_tensor_each(diff_0) * (1 - tm_low) + tm_low).detach()
        norm_ref_mask_1 = (self.norm_tensor_each(diff_1) * (1 - tm_low) + tm_low).detach()

        mask_ref_illu_list.append(norm_ref_mask_0)
        mask_ref_illu_list.append(norm_ref_mask_1)


        return mask_ref_illu_list


    def norm_tensor_each(self,tensor):
        b, channel, height, width = tensor.size()
        normed_tensor = torch.zeros(b, channel, height, width).cuda()

        for t_num in range(b):
            tensor_tmp = tensor[t_num]
            tensor_max = torch.max(tensor_tmp)
            tensor_min = torch.min(tensor_tmp)
            tensor_norm_tmp = (tensor_tmp - tensor_min) / (tensor_max - tensor_min)
            normed_tensor[t_num] = tensor_norm_tmp
        return normed_tensor




    def predict_poses_AF(self, inputs, features):
        """
            Modified from AF-SfMlearner.
            Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":

                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])

        return outputs