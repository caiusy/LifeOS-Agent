import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from lifeos_agent.router import select_tool_names
from lifeos_agent.tools import execute_tool, get_tools_by_names


DEFAULT_MINIMIND_REPO = "/Users/caius/Documents/alma/github/minimind-master"
SYSTEM_PROMPT = (
    "你是 LifeOS-Agent v0.1，一个会调用外部工具的个人知识助手。"
    "当用户问到学习进度、笔记、SFTDataset、DPO、Tool Calling、Agentic RL 时，优先调用 "
    "search_fake_obsidian。"
    "当用户问今天做什么、任务、计划时，优先调用 list_today_tasks。"
    "当用户问计算、多少、乘法、涨停价时，优先调用 calculate_math。"
    "如果用户问 A 股普通股票涨停价，默认按昨收 * 1.1 计算，并尽量保留到两位小数。"
    "不要假装已经执行了工具；先输出 <tool_call>，等拿到工具结果后再给最终回答。"
)


def parse_args():
    parser = argparse.ArgumentParser(description="LifeOS-Agent v0.1 demo")
    parser.add_argument("--prompt", type=str, default="", help="单轮测试输入；留空则进入交互模式")
    parser.add_argument("--hf_model_path", type=str, default="", help="Transformers 格式本地模型目录")
    parser.add_argument("--minimind_repo", type=str, default=DEFAULT_MINIMIND_REPO, help="MiniMind 源码仓库路径")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 路径；默认取 minimind_repo/model")
    parser.add_argument("--checkpoint_path", type=str, default="", help="MiniMind 原生 .pth 权重路径")
    parser.add_argument("--hidden_size", type=int, default=768, help="MiniMind hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="MiniMind layer count")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0, help="是否使用 MoE")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_turns", type=int, default=3)
    return parser.parse_args()


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(match.strip()))
        except Exception:
            continue
    return calls


def init_model(args):
    tokenizer_path = args.tokenizer_path or os.path.join(args.minimind_repo, "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if args.hf_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, trust_remote_code=True)
    else:
        if not args.checkpoint_path:
            raise ValueError("native MiniMind mode requires --checkpoint_path")
        sys.path.append(args.minimind_repo)
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        model = MiniMindForCausalLM(
            MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
            )
        )
        state_dict = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)

    if args.device.startswith("cuda"):
        model = model.half()
    return model.eval().to(args.device), tokenizer


def build_messages(user_input: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def generate_once(model, tokenizer, messages, tools, args):
    input_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    print("\n===== Prompt / input_text =====")
    print(input_text)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True,
    )
    return input_text, response


def run_agent(model, tokenizer, user_input: str, args):
    candidate_tool_names = select_tool_names(user_input)
    tools = get_tools_by_names(candidate_tool_names)
    messages = build_messages(user_input)

    print(f"User: {user_input}")
    print(f"Selected tools: {candidate_tool_names}")

    for turn in range(1, args.max_turns + 1):
        print(f"\n===== Turn {turn} =====")
        _, response = generate_once(model, tokenizer, messages, tools, args)
        print("\n===== Model output =====")
        print(response)

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            print("\n===== Final answer =====")
            print(response)
            return response

        print("\n===== Parsed tool_call =====")
        print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
        messages.append({"role": "assistant", "content": response})

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            result = execute_tool(tool_name, arguments)
            print("\n===== Tool result =====")
            print(f"{tool_name}: {json.dumps(result, ensure_ascii=False)}")
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    print("\n===== Stop =====")
    print(f"Reached max_turns={args.max_turns} before the model produced a final answer.")
    return ""


def main():
    args = parse_args()
    model, tokenizer = init_model(args)

    if args.prompt:
        run_agent(model, tokenizer, args.prompt, args)
        return

    while True:
        user_input = input("\nUser> ").strip()
        if not user_input:
            break
        run_agent(model, tokenizer, user_input, args)


if __name__ == "__main__":
    main()
