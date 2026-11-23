from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

from app.config import PRESETS_DIR, PROJECTS_DIR
from app.models import ChannelPreset, EditConfig, VideoProject


PROJECTS_FILE = PROJECTS_DIR / "projects.json"


class StateStore:
    def __init__(self) -> None:
        self.projects: Dict[str, VideoProject] = {}
        self.presets: Dict[str, ChannelPreset] = {}
        self.load_all()

    def load_all(self) -> None:
        self.projects.clear()
        if PROJECTS_FILE.exists():
            try:
                raw = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
                for item in raw:
                    proj = VideoProject.model_validate(item)
                    self.projects[proj.id] = proj
            except Exception:
                pass

        self.presets.clear()
        for fp in sorted(PRESETS_DIR.glob("*.json")):
            try:
                preset = ChannelPreset.model_validate_json(fp.read_text(encoding="utf-8"))
                self.presets[preset.id] = preset
            except Exception:
                continue

        # Seed default preset if none exist
        if not self.presets:
            default_p = ChannelPreset(
                id="default_review",
                name="Kênh Review Mặc Định (Thuyết minh)",
                config=EditConfig(),
            )
            self.save_preset(default_p)

    def save_projects(self) -> None:
        data = [p.model_dump() for p in self.projects.values()]
        PROJECTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_projects(self) -> List[VideoProject]:
        return list(self.projects.values())

    def get_project(self, project_id: str) -> Optional[VideoProject]:
        return self.projects.get(project_id)

    def upsert_project(self, project: VideoProject) -> VideoProject:
        self.projects[project.id] = project
        self.save_projects()
        return project

    def delete_project(self, project_id: str) -> bool:
        if project_id in self.projects:
            del self.projects[project_id]
            self.save_projects()
            return True
        return False

    def list_presets(self) -> List[ChannelPreset]:
        return list(self.presets.values())

    def save_preset(self, preset: ChannelPreset) -> ChannelPreset:
        self.presets[preset.id] = preset
        out_path = PRESETS_DIR / f"{preset.id}.json"
        out_path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")
        return preset

    def delete_preset(self, preset_id: str) -> bool:
        if preset_id in self.presets:
            del self.presets[preset_id]
            fp = PRESETS_DIR / f"{preset_id}.json"
            if fp.exists():
                fp.unlink()
            return True
        return False


store = StateStore()
