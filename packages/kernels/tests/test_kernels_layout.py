import proofnet_kernels.core
import proofnet_kernels.server


def test_packages_importable() -> None:
    assert proofnet_kernels.core.__doc__
    assert proofnet_kernels.server.__doc__
