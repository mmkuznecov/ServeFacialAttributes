from typing import Optional
import torch
from timm.data import resolve_data_config
import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import List
from PIL import Image
import numpy as np


TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from create_timm_model import create_model
else:
    from .create_timm_model import create_model

has_compile = hasattr(torch, "compile")

def prepare_batch(
    image_list: List[Image.Image],
    device: torch.device,
    target_size: int = 224,
    mean: List[float] = IMAGENET_DEFAULT_MEAN,
    std: List[float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    batch_images = []
    for img in image_list:
        img = img.resize((target_size, target_size))
        img = img.convert("RGB")
        img = torch.from_numpy(np.array(img)).float()
        img = img / 255.0
        img = img.permute(2, 0, 1)
        img = (img - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[
            :, None, None
        ]
        batch_images.append(img)

    batch_images = torch.stack(batch_images, dim=0)
    if device:
        batch_images = batch_images.to(device)
    return batch_images


def age_calculation(
    model_output: float, max_age: float, min_age: float, avg_age: float
) -> float:
    result = model_output * (max_age - min_age) + avg_age
    result = round(result, 2)
    return result


class Meta:
    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.avg_age = None
        self.num_classes = None

        self.in_chans = 3
        self.with_persons_model = False
        self.disable_faces = False
        self.use_persons = True
        self.only_age = False

        self.num_classes_gender = 2
        self.input_size = 224

    def load_from_ckpt(
        self, ckpt_path: str, disable_faces: bool = False, use_persons: bool = True
    ) -> "Meta":

        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.only_age = state["no_gender"]

        only_age = state["no_gender"]

        self.disable_faces = disable_faces
        if "with_persons_model" in state:
            self.with_persons_model = state["with_persons_model"]
        else:
            self.with_persons_model = (
                True if "patch_embed.conv1.0.weight" in state["state_dict"] else False
            )

        self.num_classes = 1 if only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError(
                "You can not disable faces and persons together. "
                "Set --with-persons if you want to run with --disable-faces"
            )
        self.input_size = state["state_dict"]["pos_embed"].shape[1] * 16
        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update(
            {
                "use_person_crops": self.use_person_crops,
                "use_face_crops": self.use_face_crops,
            }
        )
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model


class MiVOLO:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: Optional[str] = None,
    ):
        self.verbose = verbose
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        self.meta: Meta = Meta().load_from_ckpt(ckpt_path, disable_faces, use_persons)

        model_name = f"mivolo_d1_{self.meta.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            pretrained=False,
            checkpoint_path=ckpt_path,
            filter_keys=["fds."],
        )
        self.param_count = sum([m.numel() for m in self.model.parameters()])

        self.data_config = resolve_data_config(
            model=self.model,
            verbose=verbose,
            use_test_size=True,
        )

        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        self.model = self.model.to(self.device)

        if torchcompile:
            assert (
                has_compile
            ), "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)

        self.model.eval()
        if self.half:
            self.model = self.model.half()

    def warmup(self, batch_size: int, steps=10):
        if self.meta.with_persons_model:
            input_size = (6, self.input_size, self.input_size)
        else:
            input_size = self.data_config["input_size"]

        input = torch.randn((batch_size,) + tuple(input_size)).to(self.device)

        for _ in range(steps):
            out = self.inference(input)  # noqa: F841

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def inference(self, model_input: torch.tensor) -> torch.tensor:

        with torch.no_grad():
            if self.half:
                model_input = model_input.half()
            output = self.model(model_input)
        return output
