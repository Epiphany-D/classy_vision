import torch.nn as nn

from classy_vision.models import ClassyModel, register_model


@register_model("pathwayJudge")
class pathwayJudge(ClassyModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((20, 20)),
            nn.Flatten(1),
            nn.Linear(3 * 20 * 20, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls()
