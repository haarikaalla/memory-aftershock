from fastapi.testclient import TestClient

from aftershock.api import app


def test_api_ask_and_memory_listing():
    client = TestClient(app)
    memories = client.get("/memories")
    assert memories.status_code == 200
    assert any(memory["memory_id"] == "m0" for memory in memories.json())

    graph = client.get("/graph")
    assert graph.status_code == 200
    assert graph.json()["edges"] == [{"source": "m0", "target": "m1"}]

    plan = client.get("/plan/m0")
    assert plan.status_code == 200
    assert plan.json()["candidates"][0]["memory_id"] == "m1"

    answer = client.post("/ask", json={"query": "What should Maya use for restaurant plans?"})
    assert answer.status_code == 200
    assert "tool_trace" in answer.json()
