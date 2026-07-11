import json
import unittest

from lifeos_agent.main import build_messages, parse_tool_calls
from lifeos_agent.router import select_tool_names
from lifeos_agent.tools import execute_tool, get_tools_by_names


class RouterTests(unittest.TestCase):
    def test_routes_only_relevant_tool(self):
        cases = {
            "我之前学 SFTDataset 学到哪了？": ["search_fake_obsidian"],
            "我今天应该做什么？": ["list_today_tasks"],
            "17.66 涨停价是多少？": ["calculate_math"],
            "你好，简单介绍一下你自己": [],
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(select_tool_names(prompt), expected)

    def test_schema_filter_deduplicates_names(self):
        tools = get_tools_by_names(["calculate_math", "calculate_math", "missing"])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "calculate_math")


class ParserTests(unittest.TestCase):
    def test_parses_multiple_tool_calls(self):
        text = """
        <tool_call>{"name":"calculate_math","arguments":{"expression":"2+3"}}</tool_call>
        <tool_call>{"name":"list_today_tasks","arguments":{}}</tool_call>
        """
        calls = parse_tool_calls(text)
        self.assertEqual([call["name"] for call in calls], ["calculate_math", "list_today_tasks"])

    def test_ignores_invalid_json(self):
        self.assertEqual(parse_tool_calls("<tool_call>{bad json}</tool_call>"), [])


class ExecutorTests(unittest.TestCase):
    def test_dict_arguments(self):
        self.assertEqual(
            execute_tool("calculate_math", {"expression": "round(17.66 * 1.1, 2)"}),
            {"result": 19.43},
        )

    def test_json_string_arguments(self):
        self.assertEqual(execute_tool("calculate_math", '{"expression":"2+3"}'), {"result": 5})

    def test_invalid_json_arguments(self):
        self.assertEqual(execute_tool("calculate_math", "bad json"), {"error": "invalid arguments json"})

    def test_unknown_tool(self):
        self.assertEqual(execute_tool("missing", {}), {"error": "unknown tool: missing"})

    def test_non_object_arguments(self):
        self.assertEqual(
            execute_tool("calculate_math", ["2+3"]),
            {"error": "arguments must be a JSON object"},
        )

    def test_tool_result_can_be_appended_as_role_tool(self):
        messages = build_messages("17.66 涨停价是多少？")
        call_text = '<tool_call>{"name":"calculate_math","arguments":{"expression":"round(17.66*1.1,2)"}}</tool_call>'
        call = parse_tool_calls(call_text)[0]
        result = execute_tool(call["name"], call["arguments"])
        messages.append({"role": "assistant", "content": call_text})
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(json.loads(messages[-1]["content"]), {"result": 19.43})


if __name__ == "__main__":
    unittest.main()
