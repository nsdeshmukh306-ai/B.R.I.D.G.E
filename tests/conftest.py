import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def sim():
    from bridge.simulation.world import make_default_simulation

    return make_default_simulation(seed=1)


@pytest.fixture
def cam_link(sim):
    world, proj, cam = sim

    class Link:
        def capture(self, settle_s: float = 0.0):
            return cam.render()

        def size(self):
            return cam.resolution

    return Link()
