from aftershock.retrieval import MemoryRetriever


def test_retriever_uses_question_text_without_answer_labels():
    item = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What laptop does Maya use?",
        "question_date": "2025-01-03",
        "answer_session_ids": ["s2"],
        "haystack_dates": ["2025-01-01", "2025-01-02"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "Maya likes tea."}],
            [{"role": "assistant", "content": "Maya uses a linux laptop."}],
        ],
    }
    hits = MemoryRetriever.from_longmemeval_item(item).search_sessions(item["question"], top_k=1)
    assert hits[0].doc_id == "s2"
