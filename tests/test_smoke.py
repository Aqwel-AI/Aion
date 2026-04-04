"""
Smoke tests for core Aion modules (datasets, io, text, providers, former imports).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest  # pyright: ignore [reportMissingImports]


def test_batch_processor():
    from aion.datasets import batch_processor

    assert list(batch_processor([], 3)) == []
    assert list(batch_processor([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        list(batch_processor([1], 0))


def test_validate_schema():
    from aion.datasets import validate_schema

    assert validate_schema({"text": "a", "label": 1}) is True
    assert validate_schema({"text": "a"}) is False
    assert validate_schema([], required_keys=("a",)) is False
    assert validate_schema({"x": 1}, required_keys=("x",)) is True


def test_load_json_lines_file(tmp_path: Path):
    from aion.datasets import load_json_lines_file

    p = tmp_path / "t.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    rows = load_json_lines_file(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_csv_roundtrip(tmp_path: Path):
    from aion.datasets import load_csv, iter_csv_rows

    p = tmp_path / "t.csv"
    p.write_text("name,value\nfoo,1\nbar,2\n", encoding="utf-8")
    rows = load_csv(p)
    assert rows == [{"name": "foo", "value": "1"}, {"name": "bar", "value": "2"}]
    assert list(iter_csv_rows(p)) == rows


def test_text_loaders(tmp_path: Path):
    from aion.datasets import load_text, iter_text_lines

    p = tmp_path / "t.txt"
    p.write_text("line1\nline2\n", encoding="utf-8")
    assert load_text(p) == "line1\nline2\n"
    assert list(iter_text_lines(p)) == ["line1", "line2"]


def test_save_automatically_and_checksum(tmp_path: Path):
    from aion.io import save_automatically, file_sha256, verify_sha256, atomic_write

    p = tmp_path / "out.json"
    save_automatically({"k": [1, 2]}, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == {"k": [1, 2]}
    h = file_sha256(p)
    assert verify_sha256(p, h) is True
    assert verify_sha256(p, "0" * 64) is False

    t = tmp_path / "hello.txt"
    atomic_write(t, "hello")
    assert t.read_text(encoding="utf-8") == "hello"


def test_clean_text_corpus():
    from aion.text import clean_text_corpus

    assert clean_text_corpus("<div>Hi, 9!</div>") == "Hi"


def test_providers_factory():
    from aion.providers import create_provider, supported_providers

    assert "openai" in supported_providers()
    with pytest.raises(ValueError):
        create_provider("unknown_vendor")


def test_former_imports():
    from aion.former.models import Transformer
    from aion.former.training import save_transformer_weights, load_transformer_weights

    import numpy as np

    m = Transformer(vocab_size=10, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=8)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.npz"
        save_transformer_weights(m, str(path))
        m2 = Transformer(vocab_size=10, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=8)
        load_transformer_weights(m2, str(path))


def test_trainer_causal_mask_shape():
    from aion.former.models import Transformer
    from aion.former.training import Trainer
    import numpy as np

    model = Transformer(vocab_size=20, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
    tr = Trainer(model, lr=1e-3)
    x = np.zeros((2, 8), dtype=np.int64)
    y = np.zeros((2, 8), dtype=np.int64)
    loss = tr.train_step(x, y)
    assert isinstance(loss, float)


def test_parse_chat_completion_tool_calls():
    from aion.providers.structured import parse_chat_completion_response

    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                        }
                    ],
                }
            }
        ]
    }
    turn = parse_chat_completion_response(data)
    assert turn.content is None
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "add"


def test_tool_registry_and_loop():
    from aion.tools import ToolRegistry, function_tool, run_tool_loop
    from aion.testing import FakeToolProvider, make_tool_turn
    from aion.providers.structured import NormalizedToolCall

    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, required_arg_keys=["a", "b"])
    tools = [
        function_tool(
            "add",
            "add numbers",
            properties={
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            required=["a", "b"],
        )
    ]
    turns = [
        make_tool_turn(
            [NormalizedToolCall(id="c1", name="add", arguments_json='{"a":1,"b":2}')],
            content=None,
        ),
        make_tool_turn([], content="done"),
    ]
    prov = FakeToolProvider(turns)
    msgs = [{"role": "user", "content": "use add"}]
    text, out = run_tool_loop(prov, msgs, tools, reg, max_rounds=5)
    assert text == "done"
    assert any(m.get("role") == "tool" for m in out)


def test_rag_chunk_and_memory_store():
    import numpy as np

    from aion.rag import MemoryVectorStore, SimpleRAGIndex, chunk_text

    assert chunk_text("abcd", 2, 1) == ["ab", "bc", "cd"]
    store = MemoryVectorStore()
    store.add(
        ids=["a", "b"],
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        metadatas=[{"t": "x"}, {"t": "y"}],
    )
    hits = store.search(np.array([1.0, 0.0]), k=1)
    assert hits[0].id == "a"

    idx = SimpleRAGIndex(store=store, embed_fn=lambda s: np.array([1.0, 0.0] if "a" in s else [0.0, 1.0]))
    idx.index_texts(["hello a", "bye b"], chunk_size=10, overlap=0)
    q = idx.query("a", k=1)
    assert len(q) >= 1


def test_graph_dijkstra_and_components():
    from aion.algorithms import connected_components, dijkstra, shortest_path_unweighted

    g = {"A": [("B", 1.0), ("C", 4.0)], "B": [("C", 2.0)], "C": []}
    d = dijkstra(g, "A")
    assert abs(d["C"] - 3.0) < 1e-9

    ug = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
    comps = connected_components(ug)
    assert len(comps) == 1

    dg = {"A": ["B", "C"], "B": ["C"], "C": []}
    p = shortest_path_unweighted(dg, "A", "C")
    assert p == ["A", "C"] or p == ["A", "B", "C"]


def test_serialization_roundtrip(tmp_path: Path):
    from aion.serialization import read_json, write_json

    p = tmp_path / "x.json"
    write_json(p, {"x": 1, "s": "a"})
    assert read_json(p) == {"x": 1, "s": "a"}


def test_metrics_brier():
    from aion.metrics import brier_score_binary

    assert abs(brier_score_binary([0, 1], [0.1, 0.9]) - 0.01) < 1e-9


def test_packaging_read_version():
    from aion.packaging import read_version_from_init

    v = read_version_from_init()
    assert v == "0.1.9"


def test_env_dotenv(tmp_path: Path, monkeypatch):
    import os

    from aion.env import load_dotenv_file

    p = tmp_path / ".env"
    p.write_text('AION_TEST_DOTENV_KEY=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("AION_TEST_DOTENV_KEY", raising=False)
    load_dotenv_file(p, override=False)
    assert os.environ.get("AION_TEST_DOTENV_KEY") == "fromfile"
