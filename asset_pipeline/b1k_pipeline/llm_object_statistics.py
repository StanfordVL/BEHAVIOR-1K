from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    # torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ, )

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels
)

fps = 1.0

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///scr/BEHAVIOR-1K/asset_pipeline/frames/out.mp4",
                # "max_pixels": 360 * 420,
                "fps": fps,
            },
            {
                "type": "text",
                "text": "In this video, an object from a 3D simulation environment is shown in front of an (unrelated) skybox. It's a printer object. What should be the realistic dimensions of this object? Output a JSON string in the format of {'height': x, 'width': y, 'depth': z} where x, y, z are the object dimensions in centimeters.",
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    fps=fps,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
