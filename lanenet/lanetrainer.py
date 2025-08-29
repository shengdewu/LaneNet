from engine.trainer.trainer import BaseTrainer
import logging
from engine.trainer.build import BUILD_TRAINER_REGISTRY
import torch
import cv2
import numpy as np
import torchvision
from util import colors


@BUILD_TRAINER_REGISTRY.register()
class LaneTrainer(BaseTrainer):

    def __init__(self, cfg):
        super(LaneTrainer, self).__init__(cfg)
        return

    def draw_seg(self, seg_logits, seg_label, img, i):
        bs, nc = seg_logits.shape[:2]

        seg_logits = torch.argmax(seg_logits, dim=1)

        imgs = list()
        for b in range(bs):
            im = img[b].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            im = np.ascontiguousarray(im)

            lines = self.test_data_loader.dataset.seg_to_lines(seg_logits[b].numpy(), img.shape[2:], nc)
            for line in lines:
                for pt in line:
                    cv2.circle(im, pt, 5, (0, 255, 0), -1)

            imgs.append(im[:, :, ::-1])

        cv2.imwrite(f'{self.output}/{i}-seg-pts.jpg', np.concatenate(imgs, axis=1))

        seg_label = seg_label[:, 0, ...]
        fake_seg = torch.zeros_like(img)
        gt_seg = torch.zeros_like(img)
        gt_idx = torch.unique(seg_label).detach().cpu().tolist()
        fake_idx = torch.unique(seg_logits).detach().cpu().tolist()
        idx = set(gt_idx + fake_idx)
        for c in idx:
            if c == 0:
                continue
            r, g, b = colors.colors[c]['rgb']
            gt_seg[:, 0, ...][seg_label == c] = r
            fake_seg[:, 0, ...][seg_logits == c] = r
            gt_seg[:, 1, ...][seg_label == c] = g
            fake_seg[:, 1, ...][seg_logits == c] = g
            gt_seg[:, 2, ...][seg_label == c] = b
            fake_seg[:, 2, ...][seg_logits == c] = b
        img_sample = torch.cat((img, fake_seg, gt_seg), -1)
        self.save_image(img_sample, f'{self.output}/{i}-seg.jpg', False, nrow=1, normalize=False)
        return

    def draw_pts(self, pts_logits, pts_label, ims, grid_num, col_sample_step, i):
        bs, c, h, w = ims.shape

        # logits = torch.flip(logits, dims=[2])
        prob = torch.softmax(pts_logits[:, :-1, ...], dim=1)
        idx = torch.arange(1, grid_num[0] + 1).unsqueeze(0)
        idx = idx.repeat(bs, 1)
        idx = idx.reshape(bs, -1, 1, 1)
        loc = torch.sum(prob * idx, dim=1)
        logits_pts = torch.argmax(pts_logits, dim=1)
        loc[logits_pts == grid_num[0]] = 0

        imgs = list()
        for b in range(bs):
            im = ims[b].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            im = np.ascontiguousarray(im)

            lines = self.test_data_loader.dataset.to_lines(pts_label[b].numpy(), (h, w), col_sample_step[b], w)
            for line in lines:
                for pt in line:
                    cv2.circle(im, pt, 5, (0, 255, 0), -1)

            lines = self.test_data_loader.dataset.to_lines(loc[b].numpy(), (h, w), col_sample_step[b], w)
            for line in lines:
                for pt in line:
                    cv2.circle(im, pt, 5, (255, 0, 0), -1)

            imgs.append(im[:, :, ::-1])

        cv2.imwrite(f'{self.output}/{i}-pts.jpg', np.concatenate(imgs, axis=1))
        return

    def after_loop(self):
        self.model.disable_train()

        aux_acc = [0]
        acc = [0]
        for i, batch in enumerate(self.test_data_loader):
            result = self.model(batch)
            label = batch['label'].detach().cpu()

            has_aux = result.get('aux_logits', None) is not None

            img = batch['img'].mul(255).add_(0.5).clamp_(0, 255)
            col_sample_step = batch['col_sample_step']
            grid_num = batch['grid_num']
            logits = result['logits'].detach().cpu()
            acc.append(result['acc'])

            is_seg = logits.shape[2:] == img.shape[2:]
            if is_seg:
                self.draw_seg(logits, label, img, i)
                if has_aux:
                    aux_label = batch['aux_label'].detach().cpu()
                    logits_aux = result['aux_logits'].detach().cpu()
                    aux_acc.append(result['aux_acc'])
                    self.draw_pts(logits_aux, aux_label, img, grid_num, col_sample_step, i)
            else:
                self.draw_pts(logits, label, img, grid_num, col_sample_step, i)
                if has_aux:
                    aux_label = batch['aux_label'].detach().cpu()
                    logits_aux = result['aux_logits'].detach().cpu()
                    aux_acc.append(result['aux_acc'])
                    self.draw_seg(logits_aux, aux_label, img, i)

        logging.getLogger(self.default_log_name).info(
            f'test acc = {sum(acc) / len(acc)}, aux acc = {sum(aux_acc) / len(aux_acc)}')
        return

    def iterate_after(self, epoch):
        if int(epoch + 0.5) % self.checkpoint.check_period == 0:
            acc = [0.]
            aux_acc = [0.]
            self.model.disable_train()
            with torch.no_grad():
                for i, batch in enumerate(self.test_data_loader):
                    result = self.model(batch)
                    acc.append(result['acc'])
                    aux_acc.append(result.get('aux_acc', 0))
            self.model.enable_train()

            logging.getLogger(self.default_log_name).info(
                f'trainer run step {epoch} acc = {sum(acc) / len(acc)}, aux_acc = {sum(aux_acc) / len(aux_acc)}')
        return

    @torch.no_grad()
    def save_image(self, tensor: torch.Tensor, fp: str, to_int=True, **kwargs):
        grid = torchvision.utils.make_grid(tensor, **kwargs)
        # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
        if to_int:
            grid = grid.mul(255).add_(0.5).clamp_(0, 255)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)
        cv2.imwrite(fp, ndarr[:, :, ::-1])
        return
