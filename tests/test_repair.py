from aftershock.repair import RepairEngine, Verdict
from aftershock.synthetic import generate_cases, load_case_into_store


def test_recorded_repair_replaces_impacted_memory():
    case = generate_cases(n_cases=1, seed=1)[0]
    store = load_case_into_store(case, observed=False)

    def verifier(memory):
        if memory.memory_id in case.impacted_ids:
            return Verdict(status="invalid", evidence="test", replacement="repaired")
        return Verdict(status="valid", evidence="test")

    result = RepairEngine(store).repair(
        case.corrupted_root,
        corrected_root_text=case.correct_root_text,
        strategy="recorded",
        budget=2,
        verifier=verifier,
    )
    assert set(result.replaced) == set(case.impacted_ids)
    assert store.latest(case.corrupted_root).text == case.correct_root_text
