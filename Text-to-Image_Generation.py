import os
import PIL.Image
import torch
import numpy as np
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
import torchvision
import certifi

# 设置证书路径
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# specify the path to the model
model_path = "F:/PYTHON_STUDY/deepseek-ai/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = MultiModalityCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

from diffusers.models import AutoencoderKL

# remember to use bfloat16 dtype, this vae doesn't work with fp16
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae = vae.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_gen_tag


@torch.inference_mode()
def generate(
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        cfg_weight: float = 5.0,
        num_inference_steps: int = 30,
        batchsize: int = 5
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
    tokens[batchsize:, 1:] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

    # we remove the last <bog> token and replace it with t_emb later
    inputs_embeds = inputs_embeds[:, :-1, :]

    # generate with rectified flow ode
    # step 1: encode with vision_gen_enc
    z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()

    dt = 1.0 / num_inference_steps
    dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt

    # step 2: run ode
    attention_mask = torch.ones((2 * batchsize, inputs_embeds.shape[1] + 577)).to(vl_gpt.device)
    attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
    attention_mask = attention_mask.int()
    for step in range(num_inference_steps):
        # prepare inputs for the llm
        z_input = torch.cat([z, z], dim=0)  # for cfg
        t = step / num_inference_steps * 1000.
        t = torch.tensor([t] * z_input.shape[0]).to(dt)
        z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
        z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
        z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
        z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)
        llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

        # input to the llm
        # we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
        if step == 0:
            outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb,
                                                  use_cache=True,
                                                  attention_mask=attention_mask,
                                                  past_key_values=None)
            past_key_values = []
            for kv_cache in past_key_values:
                k, v = kv_cache[0], kv_cache[1]
                past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
            past_key_values = tuple(past_key_values)
        else:
            outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb,
                                                  use_cache=True,
                                                  attention_mask=attention_mask,
                                                  past_key_values=past_key_values)
        hidden_states = outputs.last_hidden_state

        # transform hidden_states back to v
        hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
        hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
        v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
        v_cond, v_uncond = torch.chunk(v, 2)
        v = cfg_weight * v_cond - (cfg_weight - 1.) * v_uncond
        z = z + dt * v

    # step 3: decode with vision_gen_dec and sdxl vae
    decoded_image = vae.decode(z / vae.config.scaling_factor).sample

    os.makedirs('generated_samples', exist_ok=True)
    save_path = os.path.join('generated_samples', "img.jpg")
    torchvision.utils.save_image(decoded_image.clip_(-1.0, 1.0) * 0.5 + 0.5, save_path)


generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
    cfg_weight=2.0,
    num_inference_steps=30,
    batchsize=5
)