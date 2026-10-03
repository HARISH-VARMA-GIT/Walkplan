import json

from models.floor_plan import FloorPlan
from modules.floor_plan_generator.svg_creator import SvgCreator


class FloorPlanRenderer:

    def __init__(self):
        self.svg_creator = SvgCreator()

    def load_plan(self, plan_path: str) -> FloorPlan:
        with open(plan_path, "r") as plan_file:
            return FloorPlan.model_validate(json.load(plan_file))

    def save_svg(self, plan: FloorPlan, svg_path: str) -> str:
        svg_text = self.svg_creator.create_svg(plan)

        with open(svg_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(svg_text)

        return svg_path

    def render_file(self, plan_path: str, svg_path: str) -> str:
        plan = self.load_plan(plan_path)
        return self.save_svg(plan, svg_path)
