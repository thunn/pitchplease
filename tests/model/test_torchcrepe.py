from pitchplease.models.torchcrepe.model import Crepe


def test_torchcrepe():
    model = Crepe()
    assert model is not None
