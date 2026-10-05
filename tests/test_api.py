from fastapi.testclient import TestClient

from aftershock.api import app


def test_api_ask_and_memory_listing():
    client = TestClient(app)
    memories = client.get("/memories")
    assert memories.status_code == 200
    assert any(memory["memory_id"] == "m0" for memory in memories.json())

    answer = client.post("/ask", json={"query": "What should Maya use for restaurant plans?"})
    assert answer.status_code == 200
    assert "tool_trace" in answer.json()
