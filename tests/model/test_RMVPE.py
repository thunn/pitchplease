from pitchplease.models.RMVPE.model import E2E


def test_RMVPE():
    # TODO: update args to be sensible values
    model = E2E(4, 1, 2, 2)
    assert model is not None
