import argparse
import warnings

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_PROMPTS = {
    "coarse": "Is the face in the image is real or spoof?",
    "attack": "What attack type of the face?",
    "reasoning": "Why is the face in the picture real or spoof?",
    "localization": "Can you locate the spoof area of the face image?",
}

SYSTEM_PREFIX = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


def load_image(path):
    image = Image.open(path)
    if len(image.getbands()) == 1:
        img = np.array(image)
        h, w = img.shape
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = (img // 1024) * 4
        arr[:, :, 1] = (img // 32) * 8
        arr[:, :, 2] = (img % 32) * 8
        image = Image.fromarray(arr, "RGB")
    return image.convert("RGB")


def build_input_ids(tokenizer, prompt, device, offset_bos=0):
    text = f"{SYSTEM_PREFIX}USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
    chunks = [tokenizer(chunk).input_ids for chunk in text.split("<image 1>\n<image 2>\n")]
    input_ids = chunks[0] + [-201] + [-202] + chunks[1][offset_bos:]
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="FaceShield single-sample demo")
    parser.add_argument("--model-path", required=True, help="Path or HF repo of merged FaceShield model")
    parser.add_argument("--rgb-image", required=True, help="Path to RGB face image")
    parser.add_argument("--sav-image", required=True, help="Path to spoof-aware auxiliary image")
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), default="coarse")
    parser.add_argument("--prompt", default=None, help="Custom prompt. Overrides --task")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings("ignore")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompt = args.prompt or TASK_PROMPTS[args.task]
    input_ids = build_input_ids(tokenizer, prompt, args.device)
    image_tensor = model.process_images(
        [load_image(args.rgb_image), load_image(args.sav_image)], model.config
    ).to(dtype=model.dtype, device=args.device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        texts=prompt,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        repetition_penalty=1.0,
    )[0]
    print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
