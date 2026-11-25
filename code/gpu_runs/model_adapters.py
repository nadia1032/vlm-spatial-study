"""Explicit family-specific loading, prompting and output decoding."""
from dataclasses import dataclass
from pathlib import Path

INSTRUCTION = 'Answer with exactly one word: yes or no.'
ADAPTER_VERSION = 'seven-model-v2'


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    family: str
    description: str
    revision: str = 'main'


MODELS = {
    'smol256': ModelSpec('HuggingFaceTB/SmolVLM2-256M-Video-Instruct', 'smol', 'SmolVLM2 256M'),
    'smol500': ModelSpec('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', 'smol', 'SmolVLM2 500M'),
    'blip': ModelSpec('Salesforce/blip-vqa-base', 'blip', 'BLIP VQA Base'),
    'internvl': ModelSpec('OpenGVLab/InternVL3-1B-hf', 'internvl', 'InternVL3 1B HF'),
    'gemma': ModelSpec('google/gemma-3-4b-it', 'gemma', 'Gemma 3 4B IT'),
    'qwen2vl': ModelSpec('Qwen/Qwen2-VL-2B-Instruct', 'qwen', 'Qwen2-VL 2B Instruct'),
    'gitvqa': ModelSpec('microsoft/git-base-vqav2', 'git', 'GIT Base VQAv2'),
}


def choose_device_dtype(device, dtype, torch):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable. Install CUDA PyTorch on a working NVIDIA GPU, or select cpu.')
    if dtype == 'auto':
        dtype = 'float32' if device == 'cpu' else ('bfloat16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'float16')
    if device == 'cpu' and dtype == 'float16':
        raise ValueError('Use float32 on CPU for this reference implementation.')
    if device == 'cuda' and dtype == 'bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('This GPU does not support native BF16. Choose an appropriate precision or another GPU.')
    return device, dtype, getattr(torch, dtype)


class ModelAdapter:
    def __init__(self, spec, revision, device, torch_dtype):
        from transformers import (AutoConfig, AutoModelForImageTextToText, AutoProcessor,
                                  BlipForQuestionAnswering, BlipProcessor, GitForCausalLM)
        self.spec, self.device, self.dtype = spec, device, torch_dtype
        # Resolve main/tag once, then use one exact commit for config, processor and weights.
        revision = revision or spec.revision
        config = AutoConfig.from_pretrained(spec.repo_id, revision=revision)
        self.resolved_revision = getattr(config, '_commit_hash', None) or revision
        processor_cls = BlipProcessor if spec.family == 'blip' else AutoProcessor
        self.processor = processor_cls.from_pretrained(spec.repo_id, revision=self.resolved_revision)
        cls = {'blip':BlipForQuestionAnswering, 'git':GitForCausalLM}.get(spec.family, AutoModelForImageTextToText)
        kwargs = dict(revision=self.resolved_revision, torch_dtype=torch_dtype,
                      low_cpu_mem_usage=True)
        # BLIP's reference eager implementation has no SDPA requirement.
        if spec.family not in ('blip','git'):
            kwargs['attn_implementation'] = 'sdpa'
        self.model = cls.from_pretrained(spec.repo_id, **kwargs).to(device).eval()

    def prompt_text(self, question):
        # BLIP is a VQA encoder/decoder, not an instruction-following chat model.
        return question if self.spec.family in ('blip','git') else question + '\n' + INSTRUCTION

    @property
    def prompt_policy(self):
        return 'spatial_question_only' if self.spec.family in ('blip','git') else 'spatial_question_plus_yes_no_instruction'

    def prepare(self, image_path, question):
        from PIL import Image
        text = self.prompt_text(question)
        if self.spec.family == 'blip':
            with Image.open(image_path) as source:
                inputs = self.processor(images=source.convert('RGB'), text=text, return_tensors='pt')
        elif self.spec.family == 'git':
            import torch
            with Image.open(image_path) as source:
                inputs = self.processor(images=source.convert('RGB'), return_tensors='pt')
            ids = self.processor.tokenizer(text, add_special_tokens=False).input_ids
            inputs['input_ids'] = torch.tensor([[self.processor.tokenizer.cls_token_id]+ids])
        elif self.spec.family == 'smol':
            # Preserve the exact two-step image/chat preparation used by the existing Smol runner.
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            with Image.open(image_path) as source:
                inputs = self.processor(text=prompt, images=[source.convert('RGB')], return_tensors='pt')
        else:
            # Native HF processor supplies family-specific image tokens, crops and masks.
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'path': str(Path(image_path).resolve())},
                {'type': 'text', 'text': text}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt')
        inputs = inputs.to(self.device)
        for name, value in inputs.items():
            if hasattr(value, 'is_floating_point') and value.is_floating_point():
                inputs[name] = value.to(dtype=self.dtype)
        return inputs

    def decode(self, generated, inputs):
        # BLIP generates decoder answers only; chat models prepend their input token sequence.
        answer_ids = generated if self.spec.family == 'blip' else generated[:, inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()

    def processor_metadata(self):
        image_processor = getattr(self.processor, 'image_processor', None)
        return dict(processor_class=type(self.processor).__name__,
                    image_processor_class=type(image_processor).__name__,
                    image_processor=image_processor.to_dict() if image_processor is not None else {})

    def predict(self, image_path, question, max_new_tokens):
        import torch
        inputs = self.prepare(image_path, question)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        return self.decode(generated, inputs), dict(
            input_tokens=int(inputs['input_ids'].shape[1]),
            generated_tokens=int(generated.shape[1]) if self.spec.family == 'blip' else int(generated.shape[1]-inputs['input_ids'].shape[1]),
            tensor_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v, 'shape')})
