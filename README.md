# FaceShield

![FaceShield Overview](images/output.jpg)

FaceShield is a face anti-spoofing (FAS) multimodal large language model built on the Bunny codebase. It supports four FAS tasks:

- Coarse-grained classification: real vs. spoof
- Fine-grained classification: attack type recognition
- Reasoning: natural-language evidence for the decision
- Attack localization: relative bounding boxes for spoof regions

## Installation

```bash
conda create -n faceshield python=3.10 -y
conda activate faceshield
pip install -r requirements.txt
pip install -e .
```

The experiments used:

- PyTorch 2.2.2 + CUDA 11.8
- Transformers 4.41.0.dev0
- SigLIP SO400M patch14 384 vision tower
- Phi-3-mini-4k-instruct language model

## Dataset Release

FaceShield data annotations are available on Hugging Face Datasets:

- https://huggingface.co/datasets/wwhhyy/FaceShield

Because face data involves privacy, security, and compliance requirements, this repository and the released annotations do not redistribute identifiable raw face images. To reproduce the experiments, please obtain the original images from the corresponding public datasets and follow their licenses, download terms, and usage restrictions:

- WMCA
- PADISI
- SiW-Mv2

We thank the authors and maintainers of WMCA, PADISI, and SiW-Mv2 for providing important resources for face anti-spoofing research.
## Model And Data Preparation

Download or prepare the following paths before running training or inference. 

```text
/path/to/Phi-3-mini-4k-instruct
/path/to/siglip-so400m-patch14-384
/path/to/pretrained/mm_projector.bin
/path/to/faceshield_train.json
/path/to/rgb_images
/path/to/spoof_aware_images
```

The base LLM, SigLIP vision tower, and projector/checkpoint preparation follow the Bunny-style workflow. If you need
projector weights or base-model setup details, refer to the Bunny project:

- https://github.com/BAAI-DCAI/Bunny

Training samples follow the Bunny multi-image instruction format:

```json
[
  {
    "id": "sample_1",
    "image": ["rgb/example.jpg", "sav/example.jpg"],
    "conversations": [
      {"from": "human", "value": "<image 1>\n<image 2>\nIs the face real or spoof?"},
      {"from": "gpt", "value": "The face is spoof."}
    ]
  }
]
```

`image[0]` is the RGB face image and `image[1]` is the SAVP auxiliary image.

## Quick Demo

```bash
python demo.py \
  --model-path /path/to/FaceShield-or-merged-checkpoint \
  --rgb-image examples/rgb.jpg \
  --sav-image examples/sav.jpg \
  --task coarse
```

Supported demo tasks:

```text
coarse       real/spoof classification
attack       attack type recognition
reasoning    explainable FAS reasoning
localization spoof region localization
```

### Task Prompt Examples

The following prompts are used by `demo.py` and can also be reused in custom evaluation scripts. The PVTM branch uses
the same prompt through `texts=prompt`.

| Task | Prompt example | Expected output style |
| --- | --- | --- |
| Coarse-grained classification | `Is the face in the image is real or spoof?` | `real` or `spoof` |
| Fine-grained attack classification | `What attack type of the face?` | attack type such as `print attack`, `replay attack`, `glasses attack`, or `real` |
| Reasoning | `Why is the face in the picture real or spoof?` | short explanation plus final decision |
| Attack localization | `Can you locate the spoof area of the face image?` | relative boxes |

Example commands:

```bash
python demo.py \
  --model-path /path/to/FaceShield-or-merged-checkpoint \
  --rgb-image /path/to/rgb.jpg \
  --sav-image /path/to/spoof_aware.jpg \
  --task coarse

python demo.py \
  --model-path /path/to/FaceShield-or-merged-checkpoint \
  --rgb-image /path/to/rgb.jpg \
  --sav-image /path/to/spoof_aware.jpg \
  --task attack

python demo.py \
  --model-path /path/to/FaceShield-or-merged-checkpoint \
  --rgb-image /path/to/rgb.jpg \
  --sav-image /path/to/spoof_aware.jpg \
  --task reasoning \
  --max-new-tokens 512

python demo.py \
  --model-path /path/to/FaceShield-or-merged-checkpoint \
  --rgb-image /path/to/rgb.jpg \
  --sav-image /path/to/spoof_aware.jpg \
  --task localization
```

You can also pass a custom prompt:

```bash
python demo.py \
  --model-path /path/to/model \
  --rgb-image examples/rgb.jpg \
  --sav-image examples/sav.jpg \
  --prompt "Can you locate the spoof area of the face image?"
```

## Training

Edit environment variables in the shell command or export them before launch:

```bash
export BASE_LLM=/path/to/Phi-3-mini-4k-instruct
export VISION_TOWER=/path/to/siglip-so400m-patch14-384
export TRAIN_JSON=/path/to/faceshield_train.json
export IMAGE_FOLDER=/path/to/image/root
export PRETRAIN_MM_PROJECTOR=/path/to/mm_projector.bin
export OUTPUT_DIR=./checkpoints-phi-3/faceshield-lora

bash script/train/finetune_lora.sh
```

For projector pretraining:

```bash
export BASE_LLM=/path/to/Phi-3-mini-4k-instruct
export VISION_TOWER=/path/to/siglip-so400m-patch14-384
export TRAIN_JSON=/path/to/faceshield_pretrain.json
export IMAGE_FOLDER=/path/to/image/root
export OUTPUT_DIR=./checkpoints-pretrain/faceshield-pretrain

bash script/train/pretrain.sh
```

## Repository Structure

```text
bunny/                    Core model, training, data, and utility code
script/deepspeed/          DeepSpeed configs
script/train/              FaceShield pretraining and LoRA SFT launch scripts
script/merge_lora_weights.py
                           LoRA merge utility
demo.py                    Single-sample FaceShield inference demo
requirements.txt           Python dependencies
```

## Citation

```bibtex
@inproceedings{wang2026faceshield,
  title={FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models},
  author={Wang, Hongyang and Shi, Yichen and Tao, Zhuofu and Gao, Yuhao and Zhang, Liepiao and Lin, Xun and Feng, Jun and Yuan, Xiaochen and Yu, Zitong and Cao, Xiaochun},
  booktitle={AAAI},
  year={2026}
}
```


