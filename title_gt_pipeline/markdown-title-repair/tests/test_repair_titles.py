from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_titles.py"
SPEC = importlib.util.spec_from_file_location("repair_titles", SCRIPT)
assert SPEC and SPEC.loader
repair_titles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair_titles)


def chat_response(result: dict[str, Any], finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": json.dumps(result, ensure_ascii=False)},
            }
        ]
    }


class FakeAPI:
    def __init__(self, responder: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                owner.requests.append(request)
                payload = owner.responder(request)
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1/chat/completions"

    def __enter__(self) -> "FakeAPI":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class RepairTitlesTest(unittest.TestCase):
    def write_config(self, root: Path, url: str) -> Path:
        (root / "rules.md").write_text("Only repair headings.\n", encoding="utf-8")
        config = {
            "api": {
                "url": url,
                "api_key_env": "",
                "model": "fake-model",
                "max_retries": 0,
            },
            "rules_file": "rules.md",
        }
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_single_file_applies_only_heading_marker_edits(self) -> None:
        source_text = (
            "Document title\n"
            "Body  with  exact spaces.\n"
            "\n"
            "# Chapter\n"
            "```python\n"
            "# code comment\n"
            "```\n"
            "## Not a heading\n"
        )
        result = {
            "edits": [
                {"line": 1, "source_line": "Document title", "level": 1, "reason": "title"},
                {"line": 4, "source_line": "# Chapter", "level": 2, "reason": "chapter"},
                {
                    "line": 8,
                    "source_line": "## Not a heading",
                    "level": 0,
                    "reason": "body",
                },
            ],
            "unresolved": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.md"
            output = root / "output.md"
            audit = root / "audit"
            source.write_text(source_text, encoding="utf-8")
            with FakeAPI(lambda _request: chat_response(result)) as api:
                config = self.write_config(root, api.url)
                summary = repair_titles.run_pipeline(source, output, config, audit)

            self.assertEqual(summary["ok"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                source_text.replace("Document title", "# Document title", 1)
                .replace("# Chapter", "## Chapter", 1)
                .replace("## Not a heading", "Not a heading", 1),
            )
            self.assertEqual(len(api.requests), 1)
            self.assertEqual(api.requests[0]["model"], "fake-model")
            user_message = json.loads(api.requests[0]["messages"][1]["content"])
            self.assertEqual(user_message["source_lines"][5], "# code comment")
            self.assertTrue((audit / "summary.json").is_file())
            self.assertTrue((audit / "results.jsonl").is_file())
            self.assertEqual(len(list((audit / "raw_responses").glob("*.json"))), 1)

    def test_fenced_edit_and_unresolved_result_fail_without_output(self) -> None:
        cases = [
            {
                "edits": [
                    {
                        "line": 3,
                        "source_line": "# still code",
                        "level": 2,
                        "reason": "invalid",
                    }
                ],
                "unresolved": [],
            },
            {"edits": [], "unresolved": [{"line": 1, "reason": "uncertain"}]},
        ]
        for index, result in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "input.md"
                output = root / "output.md"
                source.write_text(
                    "```text\n```not-a-close\n# still code\n```\n", encoding="utf-8"
                )
                with FakeAPI(lambda _request, item=result: chat_response(item)) as api:
                    config = self.write_config(root, api.url)
                    summary = repair_titles.run_pipeline(
                        source, output, config, root / "audit"
                    )
                self.assertEqual(summary["failed"], 1)
                self.assertFalse(output.exists())

    def test_indented_code_is_not_editable_and_heading_indent_is_preserved(self) -> None:
        markdown = "  Section\n    print('not a heading')\n"
        repaired = repair_titles.apply_edits(
            markdown,
            [{"line": 1, "source_line": "  Section", "level": 2, "reason": "section"}],
        )
        self.assertEqual(repaired, "  ## Section\n    print('not a heading')\n")
        with self.assertRaisesRegex(ValueError, "fenced code line"):
            repair_titles.apply_edits(
                markdown,
                [
                    {
                        "line": 2,
                        "source_line": "    print('not a heading')",
                        "level": 2,
                        "reason": "invalid",
                    }
                ],
            )

    def test_directory_batch_preserves_relative_paths(self) -> None:
        def responder(request: dict[str, Any]) -> dict[str, Any]:
            task = json.loads(request["messages"][1]["content"])
            source_line = task["source_lines"][0]
            result = {
                "edits": [
                    {"line": 1, "source_line": source_line, "level": 1, "reason": "title"}
                ],
                "unresolved": [],
            }
            return chat_response(result)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            (inputs / "nested").mkdir(parents=True)
            (inputs / "a.md").write_text("Alpha\n", encoding="utf-8")
            (inputs / "nested" / "b.md").write_text("Beta\n", encoding="utf-8")
            output = root / "output"
            with FakeAPI(responder) as api:
                config = self.write_config(root, api.url)
                summary = repair_titles.run_pipeline(inputs, output, config)
            self.assertEqual(summary["ok"], 2)
            self.assertEqual((output / "a.md").read_text(encoding="utf-8"), "# Alpha\n")
            self.assertEqual(
                (output / "nested" / "b.md").read_text(encoding="utf-8"), "# Beta\n"
            )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.md"
            output = root / "output.md"
            source.write_text("Title\n", encoding="utf-8")
            output.write_text("keep me\n", encoding="utf-8")
            config = self.write_config(root, "http://127.0.0.1:1/v1/chat/completions")
            with self.assertRaises(FileExistsError):
                repair_titles.run_pipeline(source, output, config)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")

    def test_env_name_placeholders_and_core_field_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rules.md").write_text("Rules\n", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "api": {
                            "url_env": "${CUSTOM_TITLE_URL}",
                            "model_env": "${CUSTOM_TITLE_MODEL}",
                            "api_key_env": "",
                        },
                        "rules_file": "rules.md",
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "CUSTOM_TITLE_URL": "http://127.0.0.1:9999/v1/chat/completions",
                "CUSTOM_TITLE_MODEL": "custom-model",
            }
            with patch.dict(os.environ, environment, clear=False):
                config, rules = repair_titles.load_config(config_path)
                self.assertEqual(
                    repair_titles.configured(config["api"], "url", "url_env"),
                    environment["CUSTOM_TITLE_URL"],
                )
                self.assertEqual(
                    repair_titles.configured(config["api"], "model", "model_env"),
                    "custom-model",
                )
                config["api"]["extra_body"] = {"messages": []}
                with self.assertRaisesRegex(ValueError, "cannot override core fields"):
                    repair_titles.build_request("Title\n", rules, config)

    def test_timeout_is_retried(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(chat_response({"edits": [], "unresolved": []})).encode()

        config = repair_titles.merge_config(
            repair_titles.DEFAULT_CONFIG,
            {
                "api": {
                    "url": "http://127.0.0.1:9999/v1/chat/completions",
                    "api_key_env": "",
                    "model": "fake-model",
                    "max_retries": 1,
                    "retry_backoff_seconds": 0,
                }
            },
        )
        with patch.object(
            repair_titles.urllib.request,
            "urlopen",
            side_effect=[TimeoutError("timed out"), Response()],
        ) as urlopen:
            result = repair_titles.call_api({"model": "fake-model"}, config)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
