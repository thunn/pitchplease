from pitchplease.models.FCPE.model import CFNaiveMelPE


def test_FCPE():
    model = CFNaiveMelPE(1, 1)
    assert model is not None
