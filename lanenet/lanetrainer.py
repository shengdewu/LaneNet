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

    def after_loop(self):
        self.model.disable_train()

        aux_acc = [0]
        acc = [0]
        for i, batch in enumerate(self.test_data_loader):
            result = self.model(batch)
            label = batch['label'].detach().cpu()
            aux_label = batch['aux_label'].detach().cpu()
            img = batch['img'].mul(255).add_(0.5).clamp_(0, 255)
            col_sample_step = batch['col_sample_step']
            grid_num = batch['grid_num']

            logits = result['logits'].detach().cpu()
            logits_aux = result['aux_logits'].detach().cpu()
            acc.append(result['acc'])
            aux_acc.append(result['aux_acc'])

            bs, c, h, w = img.shape
            if logits_aux.shape[2:] == img.shape[2:]:
                bs, cls_nums, anchors, nums = logits.shape
                pts_logits = logits
                pts_label = label
                seg_logits = logits_aux
                seg_label = aux_label

            else:
                bs, cls_nums, anchors, nums = logits_aux.shape
                pts_logits = logits_aux
                pts_label = aux_label
                seg_logits = logits
                seg_label = label

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
                im = img[b].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
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

            seg_logits = torch.argmax(seg_logits, dim=1)

            seg_label = seg_label[:, 0, ...]
            fake_seg = torch.zeros_like(batch['img'])
            gt_seg = torch.zeros_like(batch['img'])
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
                    aux_acc.append(result['aux_acc'])
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
