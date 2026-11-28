from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from app.config import PRESETS_DIR, PROJECTS_DIR
from app.models import ChannelPreset, EditConfig, VideoProject


PROJECTS_FILE = PROJECTS_DIR / "projects.json"
PROJECTS_BACKUP = PROJECTS_DIR / "projects.json.bak"


class StateStore:
    def __init__(self) -> None:
        # save_projects() runs from the event loop AND from worker threads (render / timing): serialize it
        self._save_lock = threading.RLock()
        self.projects: Dict[str, VideoProject] = {}
        self.presets: Dict[str, ChannelPreset] = {}
        self.load_all()

    def load_all(self) -> None:
        self.projects.clear()
        # A corrupt/partial projects.json used to silently drop EVERY project. Now: fall back to the backup,
        # and skip only the items that fail validation.
        for src in (PROJECTS_FILE, PROJECTS_BACKUP):
            if not src.exists():
                continue
            try:
                raw = json.loads(src.read_text(encoding="utf-8"))
            except Exception:
                continue
            for item in raw if isinstance(raw, list) else []:
                try:
                    proj = VideoProject.model_validate(item)
                except Exception:
                    continue
                self.projects[proj.id] = proj
            break

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
        """Atomic write (tmp file + rename) with a backup of the previous version: a crash or power loss in
        the middle of a write can no longer leave a truncated projects.json behind."""
        with self._save_lock:
            data = [p.model_dump() for p in list(self.projects.values())]
            tmp = PROJECTS_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                if PROJECTS_FILE.exists():
                    os.replace(PROJECTS_FILE, PROJECTS_BACKUP)
            except OSError:
                pass
            os.replace(tmp, PROJECTS_FILE)

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
