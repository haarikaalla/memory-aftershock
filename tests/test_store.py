import pytest

from aftershock.store import Memory, MemoryStore


def test_store_rejects_backward_edges_and_cycles():
    store = MemoryStore()
    with store.transaction():
        store.put(Memory("a", "root", "x", 1))
        store.put(Memory("b", "child", "x", 2))
        store.add_edge("a", "b")
    with pytest.raises(ValueError):
        store.add_edge("b", "a")
