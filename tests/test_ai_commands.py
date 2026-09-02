import json

import numpy as np
import pytest

from bridge.ai.base import AIError, StructuredResponseParser
from bridge.ai.mock import ScriptedMockProvider, SimulationOracleProvider, parse_intent
from bridge.ai.schemas import NormalizedBox, TargetIdentification
from bridge.interaction.commands import (
    ClearProjection, CommandPlanner, HighlightObject, PointToObject, ShowMessage, parse_command,
)


def test_parser_accepts_valid_json_and_fenced_json():
    good = {"intent": "find_object", "target": "screwdriver", "visual_description": "red handle", "confidence": 0.94,
            "boxes": [{"box_2d": [100, 200, 300, 400], "label": "screwdriver"}]}
    t = StructuredResponseParser.parse(json.dumps(good), TargetIdentification)
    assert t.target == "screwdriver" and t.confidence == 0.94 and len(t.boxes) == 1
    t2 = StructuredResponseParser.parse("Sure!\n```json\n" + json.dumps(good) + "\n```", TargetIdentification)
    assert t2.intent == "find_object"


@pytest.mark.parametrize("bad", [
    "", "not json at all", "[1,2,3]",
    json.dumps({"intent": "run_python", "target": "x"}),                    # unknown intent
    json.dumps({"intent": "find_object", "confidence": 1.7}),               # out of range
    json.dumps({"intent": "find_object", "boxes": [{"box_2d": [1, 2, 3]}]}),  # malformed box
    json.dumps({"intent": "find_object", "boxes": [{"box_2d": [900, 100, 100, 200]}]}),  # ymax < ymin
])
def test_parser_rejects_invalid_ai_output(bad):
    with pytest.raises(AIError):
        StructuredResponseParser.parse(bad, TargetIdentification)


def test_normalized_box_to_pixels():
    b = NormalizedBox(box_2d=[0, 0, 500, 1000]).to_pixels(200, 100)
    assert (b.x, b.y, b.w, b.h) == (0, 0, 200, 50)


def test_command_validation_accepts_valid_and_rejects_invalid():
    cmd = parse_command({"command": "highlight_object", "target": {"label": "screwdriver"},
                         "style": {"shape": "circle", "animation": "pulse"}})
    assert isinstance(cmd, HighlightObject)
    assert isinstance(parse_command({"command": "clear_projection"}), ClearProjection)
    with pytest.raises(ValueError):
        parse_command({"command": "execute_code", "code": "import os"})
    with pytest.raises(ValueError):
        parse_command({"command": "highlight_object", "target": {"label": "x"}, "style": {"shape": "hexagon"}})


def test_planner_low_confidence_becomes_uncertain_message():
    ident = TargetIdentification(intent="find_object", target="hammer", confidence=0.2, message="Not visible")
    cmd = CommandPlanner(min_confidence=0.5).plan(ident, 640, 480)
    assert isinstance(cmd, ShowMessage) and "uncertain" in cmd.text.lower()


def test_planner_maps_intents_and_converts_boxes():
    ident = TargetIdentification(intent="point_to_object", target="scissors", confidence=0.9,
                                 boxes=[NormalizedBox(box_2d=[0, 0, 500, 500])])
    cmd = CommandPlanner().plan(ident, 640, 480)
    assert isinstance(cmd, PointToObject)
    b = cmd.target.boxes_camera[0]
    assert (b.w, b.h) == (320, 240)
    assert isinstance(CommandPlanner().plan(TargetIdentification(intent="clear_projection", confidence=1), 1, 1), ClearProjection)


@pytest.mark.parametrize("q,intent,target", [
    ("Where is the screwdriver?", "find_object", "screwdriver"),
    ("Highlight the screwdriver", "highlight_object", "screwdriver"),
    ("Where are the screws?", "find_all", "screw"),
    ("Point to the scissors", "point_to_object", "scissors"),
    ("Highlight the red object.", "highlight_object", "red object"),
    ("clear", "clear_projection", ""),
    ("What should I pick up first?", "next_step", ""),
])
def test_mock_intent_parser(q, intent, target):
    i, t, _, _ = parse_intent(q)
    assert (i, t) == (intent, target)


def test_simulation_oracle_returns_ground_truth_boxes(sim):
    world, proj, cam = sim
    prov = SimulationOracleProvider(world, cam, latency_s=0)
    frame = cam.render()
    t = prov.identify_target(frame, "Where are the screws?")
    assert t.intent == "find_all" and len(t.boxes) == 2 and all(b.label == "screw" for b in t.boxes)
    t = prov.identify_target(frame, "Where is the hammer?")
    assert t.confidence < 0.5 and not t.boxes
    plan = prov.plan_action(frame, "assembly")
    assert plan.target == "screwdriver" and not plan.done
    assert prov.status.requests == 3


def test_scripted_mock_records_calls():
    p = ScriptedMockProvider()
    t = p.identify_target(np.zeros((10, 10, 3), np.uint8), "Where is the pen?")
    assert p.calls == ["Where is the pen?"] and t.target == "pen"


def test_gemini_provider_requires_api_key():
    from bridge.ai.gemini import GeminiProvider

    with pytest.raises(AIError):
        GeminiProvider(api_key=None)
