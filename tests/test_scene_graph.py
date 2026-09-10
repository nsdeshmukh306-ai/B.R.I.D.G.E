"""Scene graph: the persistent world model behind instant, AI-free resolution."""
import time

import numpy as np
import pytest

from bridge.perception.scene_graph import (
    SceneGraph,
    label_similarity,
    normalise_label,
    stable_gap,
)
from bridge.spatial.geometry import BoundingBox
from bridge.vision.detection import ContourDetector, Detection


def det(label, x, y, w=40, h=40, conf=0.8, source="contour"):
    return Detection.from_bbox(label, BoundingBox(x=x, y=y, w=w, h=h), conf, source)


def test_label_normalisation_and_similarity():
    assert normalise_label("the 15 Blade Scalpels!") == "15 blade scalpel"
    assert normalise_label("artery clamps") == "artery clamp"
    assert label_similarity("artery clamp", "clamp") == pytest.approx(0.5)
    assert label_similarity("mayo scissors", "Mayo Scissors") == 1.0
    assert label_similarity("scalpel", "gauze") == 0.0


def test_entity_persists_while_it_moves_and_is_not_duplicated():
    g = SceneGraph()
    g.observe([det("", 100, 100)], now=0.0)
    assert len(g) == 1
    eid = g.entities[0].id
    for i in range(1, 12):                       # drift across the table
        g.observe([det("", 100 + i * 9, 100 + i * 4)], now=i * 0.1)
    assert len(g) == 1 and g.entities[0].id == eid
    assert g.entities[0].observations == 12
    assert g.entities[0].bbox.x == pytest.approx(100 + 11 * 9, abs=2)   # tracks the latest position


def test_absence_is_remembered_not_forgotten():
    g = SceneGraph(absent_after_s=0.5)
    g.observe([det("", 100, 100), det("", 300, 100)], now=0.0)
    delta = g.observe([det("", 100, 100)], now=1.0)     # one item taken off the tray
    assert len(delta.disappeared) == 1
    assert len(g.present()) == 1 and len(g.absent()) == 1
    # ... and it comes back to the same entity, not a new one
    ids_before = {e.id for e in g.entities}
    g.observe([det("", 100, 100), det("", 302, 101)], now=1.2)
    assert {e.id for e in g.entities} == ids_before
    assert len(g.present()) == 2


def test_ai_labels_attach_to_existing_local_entities():
    g = SceneGraph()
    g.observe([det("", 100, 100), det("", 400, 200)], now=0.0)
    n = g.apply_ai([("scalpel", BoundingBox(x=102, y=98, w=42, h=42), 0.9, "steel blade"),
                    ("gauze", BoundingBox(x=398, y=202, w=40, h=40), 0.8, "white square")])
    assert n == 2 and len(g) == 2                 # labelled, not duplicated
    assert {e.label for e in g.entities} == {"scalpel", "gauze"}
    assert g.find_one("scalpel") is not None
    assert g.find_one("the scalpel please") is not None


def test_find_resolves_spoken_phrases_without_ai():
    g = SceneGraph()
    g.observe([det("", 100, 100), det("", 300, 100), det("", 500, 100)], now=0.0)
    g.apply_ai([("artery clamp", BoundingBox(x=100, y=100, w=40, h=40), 0.9, ""),
                ("artery clamp", BoundingBox(x=300, y=100, w=40, h=40), 0.9, ""),
                ("mayo scissors", BoundingBox(x=500, y=100, w=40, h=40), 0.9, "")])
    assert len(g.find("artery clamps")) == 2
    assert len(g.find("clamp")) == 2                       # partial label still matches
    assert g.find_one("scissors").label == "mayo scissors"
    assert g.find("thoracotomy retractor") == []           # no false positive
    assert g.counts_by_label()["artery clamp"] == 2


def test_conflicting_ai_labels_become_aliases_not_overwrites():
    g = SceneGraph()
    g.observe([det("", 100, 100)], now=0.0)
    box = BoundingBox(x=100, y=100, w=40, h=40)
    g.apply_ai([("needle holder", box, 0.9, "")])
    g.apply_ai([("mayo hegar", box, 0.6, "")])             # lower confidence: kept as an alias
    e = g.entities[0]
    assert e.label == "needle holder" and "mayo hegar" in e.aliases
    assert g.find_one("mayo hegar") is e                   # both names still resolve


def test_pruning_keeps_sticky_and_present_entities():
    g = SceneGraph(max_entities=5)
    for i in range(12):
        g.observe([det("", 50 * i + 10, 400)], now=float(i))
    assert len(g) <= 5
    g.clear()
    g.observe([det("", 100, 100)], now=100.0)
    g.name_entity(g.entities[0].id, "lap pad", tags=["sponge"], sticky=True)
    for i in range(20):
        g.observe([det("", 60 * (i % 8) + 20, 600)], now=200.0 + i)
    assert any(e.sticky and e.label == "lap pad" for e in g.entities)


def test_scene_graph_on_the_simulated_bench(sim):
    """Against the simulator, the graph should see the real objects and keep them stable."""
    world, proj, cam = sim
    g = SceneGraph()
    d = ContourDetector()
    frame = cam.render()
    g.observe(d.detect(frame), frame, now=0.0)
    assert len(g.present()) >= 5
    first = len(g)
    for i in range(1, 10):                       # nothing changes on the bench
        f = cam.render()
        g.observe(d.detect(f), f, now=i * 0.2)
    assert len(g) == pytest.approx(first, abs=2)  # stable: no runaway entity creation

    # The screwdriver's colour is learned locally, so a colour phrase resolves it.
    gt = cam.object_bbox_in_camera(world.get("obj-screwdriver"))
    g.apply_ai([("screwdriver", gt, 0.9, "red handled")])
    hit = g.find_one("screwdriver")
    assert hit is not None and hit.center.distance_to(gt.center) < 40


def test_gap_polygon_is_centred_on_the_last_known_position():
    g = SceneGraph()
    g.observe([det("", 200, 200, 60, 60)], now=0.0)
    e = g.entities[0]
    poly = stable_gap(e)
    xs = [p.x for p in poly]
    ys = [p.y for p in poly]
    assert len(poly) == 4
    assert sum(xs) / 4 == pytest.approx(e.center.x, abs=1)
    assert sum(ys) / 4 == pytest.approx(e.center.y, abs=1)
    assert max(xs) - min(xs) >= 120


def test_colour_is_read_from_the_frame():
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[90:130, 90:130] = (0, 0, 220)          # BGR red patch
    g = SceneGraph()
    g.observe([det("", 90, 90, 40, 40)], frame, now=0.0)
    assert g.entities[0].color == "red"
