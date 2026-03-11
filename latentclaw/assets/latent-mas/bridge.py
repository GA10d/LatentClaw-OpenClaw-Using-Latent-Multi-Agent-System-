#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Optional


def load_modules():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "LatentMAS bridge requires `torch` and `transformers` in the active Python environment."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def normalize_device(torch_mod, raw_device: Optional[str]) -> str:
    if raw_device and raw_device.strip():
        return raw_device.strip()
    return "cuda" if torch_mod.cuda.is_available() else "cpu"


def resolve_torch_dtype(torch_mod, device: str, raw_dtype: str):
    key = (raw_dtype or "auto").strip().lower()
    if key == "bfloat16":
        return torch_mod.bfloat16
    if key == "float16":
        return torch_mod.float16
    if key == "float32":
        return torch_mod.float32
    if device.startswith("cuda"):
        return torch_mod.bfloat16
    return torch_mod.float32


def past_length(past_key_values) -> int:
    if not past_key_values:
        return 0
    first = past_key_values[0]
    if isinstance(first, tuple):
        return int(first[0].shape[-2])
    return 0


class ModelWrapper:
    def __init__(self, model_name: str, device: str, trust_remote_code: bool, torch_dtype: str):
        self.torch, AutoModelForCausalLM, AutoTokenizer = load_modules()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        ensure_pad_token(self.tokenizer)
        dtype = resolve_torch_dtype(self.torch, device, torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self._latent_realign_matrix = None

    def render_chat(self, messages: List[Dict[str, str]]) -> str:
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            rendered.append(f"<|{role}|>\n{content}\n</|{role}|>")
        rendered.append("<|assistant|>")
        return "\n".join(rendered)

    def encode_prompt(self, prompt: str):
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def _ensure_latent_realign_matrix(self):
        if self._latent_realign_matrix is not None:
            return self._latent_realign_matrix
        input_embeds = self.model.get_input_embeddings()
        output_embeds = self.model.get_output_embeddings()
        if output_embeds is None:
            output_embeds = getattr(self.model, "lm_head", None)
        if input_embeds is None or output_embeds is None:
            raise RuntimeError("Model embeddings are not accessible for latent realignment.")
        input_weight = input_embeds.weight.detach().to(device=self.device, dtype=self.torch.float32)
        output_weight = output_embeds.weight.detach().to(device=self.device, dtype=self.torch.float32)
        gram = self.torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * self.torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = self.torch.matmul(output_weight.T, input_weight)
        matrix = self.torch.linalg.solve(gram + reg, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()
        self._latent_realign_matrix = (matrix, target_norm)
        return self._latent_realign_matrix

    def apply_latent_realignment(self, hidden, enabled: bool):
        if not enabled:
            return hidden
        matrix, target_norm = self._ensure_latent_realign_matrix()
        hidden_fp32 = hidden.to(self.torch.float32)
        aligned = self.torch.matmul(hidden_fp32, matrix)
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def generate_latent(
        self,
        prompt: str,
        latent_steps: int,
        past_key_values,
        think_token: bool,
        latent_space_realign: bool,
    ):
        wrapped_prompt = f"{prompt}<think>" if think_token else prompt
        input_ids, attention_mask = self.encode_prompt(wrapped_prompt)
        prompt_tokens = int(attention_mask.sum().item())

        if past_key_values is not None:
            past_len = past_length(past_key_values)
            if past_len > 0:
                past_mask = self.torch.ones((1, past_len), dtype=attention_mask.dtype, device=self.device)
                attention_mask = self.torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        for _ in range(max(0, latent_steps)):
            latent_vec = self.apply_latent_realignment(last_hidden, latent_space_realign)
            latent_embed = latent_vec.unsqueeze(1)
            latent_mask = self.torch.ones(
                (latent_embed.shape[0], past_length(past) + 1),
                dtype=self.torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past, prompt_tokens

    def generate_text(
        self,
        prompt: str,
        past_key_values,
        think_token: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        wrapped_prompt = f"{prompt}<think>" if think_token else prompt
        input_ids, attention_mask = self.encode_prompt(wrapped_prompt)
        prompt_tokens = int(attention_mask.sum().item())

        cache_position = None
        if past_key_values is not None:
            past_len = past_length(past_key_values)
            cache_position = self.torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=self.torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = self.torch.ones((1, past_len), dtype=attention_mask.dtype, device=self.device)
                attention_mask = self.torch.cat([past_mask, attention_mask], dim=-1)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
        }
        if temperature > 0:
            generate_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        outputs = self.model.generate(**generate_kwargs)
        generated_ids = outputs.sequences[0, prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text, prompt_tokens, int(generated_ids.shape[-1])


def default_agents():
    return [
        ("Planner", "planner"),
        ("Critic", "critic"),
        ("Refiner", "refiner"),
        ("Judger", "judger"),
    ]


def compose_task_text(prompt: str, session_context: str, extra_system_prompt: str) -> str:
    sections = []
    if extra_system_prompt:
        sections.append("System guidance:\n" + extra_system_prompt.strip())
    if session_context:
        sections.append("Recent conversation:\n" + session_context.strip())
    sections.append("Current user request:\n" + prompt.strip())
    return "\n\n".join(section for section in sections if section.strip())


def build_sequential_prompt(role: str, task_text: str) -> List[Dict[str, str]]:
    system_message = "You are a local reasoning assistant participating in a latent multi-agent workflow."

    if role == "planner":
        user_prompt = (
            "You are the Planner agent.\n"
            "Read the task and produce a concise step-by-step plan.\n"
            "Do not write the final answer.\n\n"
            f"{task_text}\n\n"
            "Your plan:"
        )
    elif role == "critic":
        user_prompt = (
            "You are the Critic agent.\n"
            "You receive the previous agent's reasoning in latent form.\n"
            "Review the task, reconstruct the likely plan, and identify weaknesses or missing checks.\n"
            "Do not write the final answer.\n\n"
            f"{task_text}\n\n"
            "Your critique:"
        )
    elif role == "refiner":
        user_prompt = (
            "You are the Refiner agent.\n"
            "You receive prior reasoning and critique in latent form.\n"
            "Produce a stronger plan or solution outline that fixes likely issues.\n"
            "Do not write the final answer.\n\n"
            f"{task_text}\n\n"
            "Your refined reasoning:"
        )
    elif role == "judger":
        user_prompt = (
            "You are the Judger agent.\n"
            "You receive previous agents' reasoning in latent form.\n"
            "Use it only if helpful, and answer the user's request directly.\n"
            "Return only the final assistant response.\n\n"
            f"{task_text}\n\n"
            "Final response:"
        )
    else:
        raise ValueError(f"Unsupported role: {role}")

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_hierarchical_prompt(role: str, task_text: str) -> List[Dict[str, str]]:
    system_message = "You are a local reasoning assistant participating in a latent multi-agent workflow."
    role_map = {
        "planner": "Reason about the task from a planning and decomposition perspective.",
        "critic": "Reason about the task from a verification and risk-checking perspective.",
        "refiner": "Reason about the task from an implementation and concrete execution perspective.",
        "judger": "Synthesize the latent reasoning into the final assistant response.",
    }
    user_prompt = (
        f"{role_map[role]}\n"
        "You receive other agent reasoning in latent form when available.\n"
        "Return only content appropriate for your role.\n\n"
        f"{task_text}\n\n"
        f"{role.title()} output:"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_prompt(prompt_mode: str, role: str, task_text: str) -> List[Dict[str, str]]:
    if prompt_mode == "hierarchical":
        return build_hierarchical_prompt(role, task_text)
    return build_sequential_prompt(role, task_text)


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    config = payload.get("config") or {}
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing prompt.")

    torch_mod, _, _ = load_modules()
    device = normalize_device(torch_mod, config.get("device"))
    model_name = str(config.get("hfModelName") or "").strip()
    if not model_name:
        raise ValueError("Missing hfModelName.")

    wrapper = ModelWrapper(
        model_name=model_name,
        device=device,
        trust_remote_code=bool(config.get("trustRemoteCode", False)),
        torch_dtype=str(config.get("torchDtype") or "auto"),
    )
    task_text = compose_task_text(
        prompt=prompt,
        session_context=str(payload.get("sessionContext") or "").strip(),
        extra_system_prompt=str(payload.get("extraSystemPrompt") or "").strip(),
    )
    latent_steps = int(config.get("latentSteps") or 0)
    prompt_mode = str(config.get("prompt") or "sequential").strip().lower()
    think_token = bool(config.get("thinkToken", False))
    latent_space_realign = bool(config.get("latentSpaceRealign", True))
    max_new_tokens = int(config.get("maxNewTokens") or 1024)
    temperature = float(config.get("temperature") or 0.6)
    top_p = float(config.get("topP") or 0.95)

    total_input_tokens = 0
    total_output_tokens = 0
    past = None

    with wrapper.torch.no_grad():
        for _, role in default_agents():
            rendered = wrapper.render_chat(build_agent_prompt(prompt_mode, role, task_text))
            if role == "judger":
                text, prompt_tokens, output_tokens = wrapper.generate_text(
                    rendered,
                    past_key_values=past,
                    think_token=think_token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                total_input_tokens += prompt_tokens
                total_output_tokens += output_tokens
                print(
                    json.dumps(
                        {
                            "text": text,
                            "usage": {
                                "input": total_input_tokens,
                                "output": total_output_tokens,
                                "total": total_input_tokens + total_output_tokens,
                            },
                        }
                    )
                )
                return 0

            past, prompt_tokens = wrapper.generate_latent(
                rendered,
                latent_steps=latent_steps,
                past_key_values=past,
                think_token=think_token,
                latent_space_realign=latent_space_realign,
            )
            total_input_tokens += prompt_tokens

    raise RuntimeError("LatentMAS bridge exited without producing a final response.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
