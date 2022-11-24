from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=450,
        center_crop=False,
        length=None,
    ):
        self.size = size  # image size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # check if image folder exist
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        # image names
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        # number of images
        self.num_instance_images = len(self.instance_images_path)
        # text describing image
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        # Prior examples handling
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
            self.class_prompt_ids = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        else:
            self.class_data_root = None

        if length:
            self._length = length

        # image transform
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.instance_prompt_ids = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        print("a")
        example = {}
        # load exaples of my concept
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        print("b")

        # transfor images
        example["instance_images"] = self.image_transforms(instance_image)
        # tokenize text prompt
        example["instance_prompt_ids"] = self.instance_prompt_ids

        # handle prior examples
        if self.class_data_root:
            # load prior examples
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            # make it RGB
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            # transform
            example["class_images"] = self.image_transforms(class_image)
            # tokenize description
            example["class_prompt_ids"] = self.class_prompt_ids

        print("c")

        return example


# data set of the text
class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
