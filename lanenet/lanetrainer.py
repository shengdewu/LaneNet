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

        seg_acc = [0]
        cls_acc = [0]
        for i, batch in enumerate(self.test_data_loader):
            result = self.model(batch)
            cls = batch['label'].detach().cpu()
            seg = batch['aux_label'].detach().cpu()
            img = batch['img'].mul(255).add_(0.5).clamp_(0, 255)
            col_sample_step = batch['col_sample_step']
            grid_num = batch['grid_num']

            logits = result['logits'].detach().cpu()
            logits_seg = result['aux_logits'].detach().cpu()
            cls_acc.append(result['acc'])
            seg_acc.append(result['aux_acc'])

            bs, c, h, w = img.shape
            bs, cls_nums, anchors, nums = logits.shape

            # logits = torch.flip(logits, dims=[2])
            prob = torch.softmax(logits[:, :-1, ...], dim=1)
            idx = torch.arange(1, grid_num[0] + 1).unsqueeze(0)
            idx = idx.repeat(bs, 1)
            idx = idx.reshape(bs, -1, 1, 1)
            loc = torch.sum(prob * idx, dim=1)
            logits = torch.argmax(logits, dim=1)
            loc[logits == grid_num[0]] = 0

            imgs = list()
            for b in range(bs):
                im = img[b].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                im = np.ascontiguousarray(im)

                pts = self.test_data_loader.dataset.to_pts(cls[b].numpy(), (h, w), col_sample_step[b])
                for pt in pts:
                    cv2.circle(im, pt, 5, (0, 255, 0), -1)

                pts = self.test_data_loader.dataset.to_pts(loc[b].numpy(), (h, w), col_sample_step[b])
                for pt in pts:
                    cv2.circle(im, pt, 5, (255, 0, 0), -1)

                imgs.append(im[:, :, ::-1])

            cv2.imwrite(f'{self.output}/{i}-lane.jpg', np.concatenate(imgs, axis=1))

            logits_seg = torch.argmax(logits_seg, dim=1)

            seg = seg[:, 0, ...]
            fake_seg = torch.zeros_like(batch['img'])
            gt_seg = torch.zeros_like(batch['img'])
            gt_idx = torch.unique(seg).detach().cpu().tolist()
            fake_idx = torch.unique(logits_seg).detach().cpu().tolist()
            idx = set(gt_idx + fake_idx)
            for c in idx:
                if c == 0:
                    continue
                r, g, b = colors.colors[c]['rgb']
                gt_seg[:, 0, ...][seg == c] = r
                fake_seg[:, 0, ...][logits_seg == c] = r
                gt_seg[:, 1, ...][seg == c] = g
                fake_seg[:, 1, ...][logits_seg == c] = g
                gt_seg[:, 2, ...][seg == c] = b
                fake_seg[:, 2, ...][logits_seg == c] = b
            img_sample = torch.cat((img, fake_seg, gt_seg), -1)
            self.save_image(img_sample, f'{self.output}/{i}-seg.jpg', False, nrow=1, normalize=False)

        logging.getLogger(self.default_log_name).info(
            f'test cls acc = {sum(cls_acc) / len(cls_acc)}, seg acc = {sum(cls_acc) / len(cls_acc)}')
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
                f'trainer run step {epoch} acc = {sum(acc) / len(acc)}, seg = {sum(aux_acc) / len(aux_acc)}')
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
