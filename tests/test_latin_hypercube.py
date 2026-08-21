from farpoint.variation_engine import DeterministicLatinHypercubeSampler


def test_latin_hypercube_is_continuous_deterministic_and_covers_each_stratum():
    sampler = DeterministicLatinHypercubeSampler(
        bounds=(("x", 0.0, 1.0), ("yaw", 0.0, 90.0)), population=10, seed=17
    )
    samples = [sampler.sample(slot) for slot in range(10)]
    assert samples == [sampler.sample(slot) for slot in range(10)]
    assert {sample["strata"]["x"] for sample in samples} == set(range(10))
    assert {sample["strata"]["yaw"] for sample in samples} == set(range(10))
    assert all(0.0 <= sample["values"]["x"] < 1.0 for sample in samples)
    assert all(0.0 <= sample["values"]["yaw"] < 90.0 for sample in samples)
    replacement = DeterministicLatinHypercubeSampler(
        bounds=(("x", 0.0, 1.0), ("yaw", 0.0, 90.0)), population=10, seed=99
    ).sample_in_strata(samples[3]["strata"], slot=3)
    assert replacement["strata"] == samples[3]["strata"]
    assert replacement["values"] != samples[3]["values"]
