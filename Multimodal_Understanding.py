import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# specify the path to the model
model_path = "F:/PYTHON_STUDY/deepseek-ai/Janus-Pro-7B"
# 修改这里，使用 fast_image_processor_class 并移除弃用的参数
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
    model_path,
    fast_image_processor_class="SomeClass",  # 这里需要根据实际情况替换成正确的类名
    # use_fast=True  # 移除这个参数，避免警告
)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 改用float16节省显存
    device_map="auto"  # 自动分配设备
)
# 尝试使用 float16 而不是 bfloat16，避免显存问题
# vl_gpt = vl_gpt.to(torch.float16).cuda().eval()
vl_gpt = vl_gpt.to(torch.float16).eval()  # 在 CPU 上运行

# 定义 question 变量，这里你可以替换成具体的问题
question = "这张图片里有什么东西？"

# 定义 image 变量，这里你需要替换成实际的图片路径
image = "C:/Users/15857/Pictures/Warframe/Warframe0002.jpg"
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# 将 prepare_inputs 中的所有张量转换为 float16 类型，但排除 input_ids
for key in dir(prepare_inputs):
    if not key.startswith("__") and key != "input_ids":
        value = getattr(prepare_inputs, key)
        if isinstance(value, torch.Tensor):
            setattr(prepare_inputs, key, value.to(torch.float16))

# 确保索引张量是布尔类型
index_mask_keys = ['images_seq_mask', 'images_emb_mask']  # 根据实际情况调整
for key in index_mask_keys:
    if hasattr(prepare_inputs, key):
        mask = getattr(prepare_inputs, key)
        if isinstance(mask, torch.Tensor) and mask.dtype not in [torch.bool, torch.long, torch.int]:
            setattr(prepare_inputs, key, mask.to(torch.bool))

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs.__dict__)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs.sft_format[0]}", answer)