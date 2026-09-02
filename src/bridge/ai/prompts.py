"""Prompt construction for the vision/reasoning model."""
from __future__ import annotations

INTENT_LIST = (
    "find_object, find_all, highlight_object, point_to_object, label_object, show_target_zone, "
    "draw_path, show_message, clear_projection, describe_scene, next_step, unknown"
)

SYSTEM_CONTEXT = (
    "You are the reasoning module of BRIDGE, a system that projects graphics onto a real physical "
    "surface (a table or wall) seen by a fixed camera. You receive the camera image and a user "
    "request. You never execute actions; you return a structured JSON decision that the application "
    "executes. Be precise about object locations and honest about uncertainty."
)


class PromptBuilder:
    @staticmethod
    def identify_target(user_query: str, width: int, height: int, known_labels: list[str] | None = None) -> str:
        known = ""
        if known_labels:
            known = f"\nObjects currently tracked locally: {', '.join(known_labels)}."
        return (
            f"{SYSTEM_CONTEXT}\n\n"
            f"User request: \"{user_query}\"\n"
            f"The image is {width}x{height} pixels.{known}\n\n"
            "Return JSON with fields:\n"
            f"- intent: one of [{INTENT_LIST}]\n"
            "- target: short noun for the requested object (e.g. 'screwdriver'), or '' if none\n"
            "- visual_description: distinctive appearance (colour, shape) of the target in THIS image\n"
            "- confidence: 0..1 that you found the right object(s)\n"
            "- boxes: list of {box_2d:[ymin,xmin,ymax,xmax] on a 0-1000 scale, label} for EVERY matching object "
            "(use find_all when the user asks for all/plural instances). Empty list if not visible.\n"
            "- secondary_target and secondary_boxes: for draw_path/show_target_zone, the destination.\n"
            "- style: circle | outline | point; animation: pulse | blink | none\n"
            "- message: short text to show the user if uncertain, or a one-line answer\n"
            "If the object is not visible, or several candidates are equally plausible, set confidence below 0.5 "
            "and explain in message."
        )

    @staticmethod
    def understand_scene(width: int, height: int) -> str:
        return (
            f"{SYSTEM_CONTEXT}\n\nList every distinct physical object on the surface in this {width}x{height} image. "
            "Return JSON {objects:[{label, visual_description, box:{box_2d:[ymin,xmin,ymax,xmax], label}, confidence}], "
            "summary}. box_2d uses a 0-1000 scale."
        )

    @staticmethod
    def plan_action(task_context: str, width: int, height: int) -> str:
        return (
            f"{SYSTEM_CONTEXT}\n\nTask context: {task_context}\n"
            f"Looking at this {width}x{height} image, decide the single next physical action for the user.\n"
            "Return JSON {instruction, target, visual_description, boxes:[{box_2d:[ymin,xmin,ymax,xmax], label}], "
            "confidence, done}. done=true if the task is complete."
        )
