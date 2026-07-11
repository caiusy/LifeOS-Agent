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
    "如果当前提供了工具，并且问题需要工具，请先输出 <tool_call>，"
    "等拿到 tool result 后再结合结果给出自然、简洁的最终回答。"
    "如果问题不需要工具，就直接正常回答，不要臆造工具调用。"
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
    if args.hf_model_path:
        tokenizer_path = args.tokenizer_path or args.hf_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, trust_remote_code=True)
    else:
        tokenizer_path = args.tokenizer_path or os.path.join(args.minimind_repo, "model")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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


def _looks_degenerate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) > 350:
        return True
    bad_markers = [
        "如果当前提供了工具",
        "按昨收 * 1.1",
        "find_math",
    ]
    if any(marker in stripped for marker in bad_markers):
        return True
    if stripped.count("* 1.1") >= 2:
        return True
    if stripped.count("不需要直接使用外部工具") >= 2:
        return True
    if stripped.count("SFTDataset") >= 2 and stripped.count("跑通") >= 2:
        return True
    return False


def render_fallback_answer(tool_name: str, result: dict, user_input: str) -> str:
    if tool_name == "list_today_tasks":
        tasks = result.get("tasks", [])
        if tasks:
            return f"你今天建议按这个顺序推进：先{tasks[0]}，再{tasks[1]}，最后{tasks[2]}。"
    if tool_name == "calculate_math":
        if "result" in result:
            return f"{user_input.replace('？', '').replace('?', '')} 的结果是 {result['result']}。"
    if tool_name == "search_fake_obsidian":
        results = result.get("results", [])
        if results:
            top = results[0]
            return f"我查到最相关的是《{top['title']}》：{top['content']}"
    return ""


def render_no_tool_fallback(user_input: str) -> str:
    text = user_input.strip()
    if any(key in text for key in ["介绍一下你自己", "你是谁", "简单介绍"]):
        return "我是 LifeOS-Agent v0.1，一个基于 MiniMind Tool Calling 外部循环搭起来的个人知识助手。当前版本会在需要时调用笔记检索、今日任务和简单计算工具。"
    if "你能做哪些事" in text:
        return "当前我主要能做三类事：检索 fake Obsidian 笔记、查看今日任务，以及处理简单数学表达式。重点是验证 Tool Calling 外部循环。"
    return ""


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
    tools = get_tools_by_names(candidate_tool_names) or None
    messages = build_messages(user_input)
    last_tool_result = None
    last_tool_name = ""

    print(f"User: {user_input}")
    print(f"Selected tools: {candidate_tool_names}")

    for turn in range(1, args.max_turns + 1):
        print(f"\n===== Turn {turn} =====")
        _, response = generate_once(model, tokenizer, messages, tools, args)
        print("\n===== Model output =====")
        print(response)

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            if last_tool_name and last_tool_result and _looks_degenerate(response):
                fallback = render_fallback_answer(last_tool_name, last_tool_result, user_input)
                if fallback:
                    print("\n===== Fallback answer =====")
                    print(fallback)
                    print("\n===== Final answer =====")
                    print(fallback)
                    return fallback
            if not last_tool_name and _looks_degenerate(response):
                fallback = render_no_tool_fallback(user_input)
                if fallback:
                    print("\n===== Fallback answer =====")
                    print(fallback)
                    print("\n===== Final answer =====")
                    print(fallback)
                    return fallback
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
            last_tool_name = tool_name
            last_tool_result = result
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
