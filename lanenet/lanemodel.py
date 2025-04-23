from engine.model.base_model import BaseModel
from engine.model.build import BUILD_MODEL_REGISTRY


@BUILD_MODEL_REGISTRY.register()
class LaneModel(BaseModel):
    def __init__(self, cfg):
        super(LaneModel, self).__init__(cfg)
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        此方法必须实现
        """
        label = dict(
            label=data['label'].to(self.device, non_blocking=True),
            aux_label=data['aux_label'].to(self.device, non_blocking=True)
        )

        result = self.g_model.forward_train(data['img'].to(self.device, non_blocking=True), label)
        return result

    def generator(self, data):
        """
        此方法必须实现
        """
        img = data['img'].to(self.device, non_blocking=True)
        label = dict(
            label=data['label'].to(self.device, non_blocking=True),
            aux_label=data['aux_label'].to(self.device, non_blocking=True)
        )
        result = self.g_model.forward_test(img, label)
        return result
