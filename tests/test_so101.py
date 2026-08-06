import numpy as np
import pytest

from farpoint.so101 import (
    LEROBOT_JOINT_NAMES,
    SIM_JOINT_NAMES,
    lerobot_to_radians,
    mapping_metadata,
    radians_to_lerobot,
)


def test_so101_joint_mapping_round_trip():
    values = np.asarray([-80.0, -25.0, 35.0, 60.0, -10.0, 75.0], dtype=np.float32)
    recovered = radians_to_lerobot(lerobot_to_radians(values))
    np.testing.assert_allclose(recovered, values, atol=1e-4)


def test_so101_mapping_rejects_bad_shape_and_range():
    with pytest.raises(ValueError, match="shape"):
        lerobot_to_radians([0.0] * 5)
    with pytest.raises(ValueError, match="outside"):
        lerobot_to_radians([101.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_so101_mapping_metadata_declares_both_conventions():
    metadata = mapping_metadata()
    assert metadata["sim_joint_names"] == list(SIM_JOINT_NAMES)
    assert metadata["feature_names"] == list(LEROBOT_JOINT_NAMES)
    assert metadata["source_unit"] == "radian"
