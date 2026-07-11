import ast
import json
import math

from lifeos_agent.fake_notes import search_notes


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": (
                "计算简单数学表达式。适合处理加减乘除、括号、幂运算，以及类似涨停价这样的数值问题。"
                "如果是 A 股普通股票涨停价，可先按昨收 * 1.1 构造表达式。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 256 * 37 或 round(17.66 * 1.1, 2)",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_today_tasks",
            "description": "列出当前固定的今日任务清单。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_fake_obsidian",
            "description": "在内置 fake Obsidian 笔记中搜索相关内容，返回最相关的 3 条笔记。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词或问题，例如 SFTDataset、Tool Calling、DPO。",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {tool["function"]["name"]: tool for tool in TOOLS}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "pow": pow,
    "min": min,
    "max": max,
}

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def get_tools_by_names(names: list[str]) -> list[dict]:
    """按路由结果筛选 schema，避免每轮把全部工具塞进 prompt。"""
    seen = set()
    selected = []
    for name in names:
        if name in TOOL_MAP and name not in seen:
            selected.append(TOOL_MAP[name])
            seen.add(name)
    return selected


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _ALLOWED_FUNCTIONS.get(node.func.id)
        if not fn:
            raise ValueError(f"unsupported function: {node.func.id}")
        args = [_eval_ast(arg) for arg in node.args]
        return fn(*args)
    raise ValueError("unsupported expression")


def safe_calculate(expression: str):
    """用 AST 白名单计算表达式，不执行任意 Python 代码。"""
    tree = ast.parse(expression, mode="eval")
    result = _eval_ast(tree)
    if isinstance(result, float):
        result = round(result, 6)
        if result.is_integer():
            result = int(result)
    return result


def calculate_math(arguments: dict) -> dict:
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        return {"error": "missing expression"}
    return {"result": safe_calculate(expression)}


def list_today_tasks(_: dict | None = None) -> dict:
    return {
        "tasks": [
            "整理 Tool Calling 笔记",
            "复习 SFTDataset",
            "跑通 LifeOS-Agent v0.1",
        ]
    }


def search_fake_obsidian(arguments: dict) -> dict:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"error": "missing query"}
    return {"results": search_notes(query, limit=3)}


def execute_tool(name: str, arguments) -> dict:
    """统一工具边界：规范化参数、校验工具名、捕获执行异常。

    训练/推理中 arguments 可能是 dict，也可能是模型输出的 JSON string，
    因此不能直接把 arguments 传给 handler。
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return {"error": "invalid arguments json"}
    elif arguments is None:
        arguments = {}
    elif not isinstance(arguments, dict):
        return {"error": "arguments must be a JSON object"}

    handlers = {
        "calculate_math": calculate_math,
        "list_today_tasks": list_today_tasks,
        "search_fake_obsidian": search_fake_obsidian,
    }
    handler = handlers.get(name)
    if not handler:
        return {"error": f"unknown tool: {name}"}
    try:
        return handler(arguments or {})
    except Exception as exc:
        return {"error": f"tool execution failed: {exc}"}
