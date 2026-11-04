"""Seo-only 20-DoF K1; retain the original leg actuator/control settings."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from booster_assets import BOOSTER_ASSETS_DIR
from JY_train.assets.robots.booster import BOOSTER_K1_CFG
from JY_train.assets.robots.actuator import BoosterDelayedPDActuatorCfg, BoosterJointR14
from .joint_config import K1_ARM_DEFAULT_POS, K1_ARM_JOINT_NAMES, K1_WALK_JOINT_NAMES

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def prepare_urdf() -> str:
    """Resolve meshes without altering the original/shared robot asset.

    SEO_K1_MESH_DIR is a directory containing the team's K1 STL files.
    Otherwise use bundled files then the installed booster_assets mesh directory.
    Only a resolved URDF is cached in the OS temporary directory.
    """
    tree = ET.parse(ASSET_DIR / "K1_20dof.urdf")
    root = tree.getroot()
    movable = {j.attrib["name"] for j in root.findall("joint")
               if j.attrib["type"] != "fixed"}
    if movable != set(K1_WALK_JOINT_NAMES):
        raise ValueError(f"Expected 20 K1 limb joints, found: {sorted(movable)}")
    for name in ("AAHead_yaw", "Head_pitch"):
        joint = root.find(f"./joint[@name='{name}']")
        if joint is None or joint.attrib["type"] != "fixed":
            raise ValueError(f"Head joint must be fixed: {name}")

    override = os.environ.get("SEO_K1_MESH_DIR")
    search_dirs = ([Path(override).expanduser()] if override else [
        ASSET_DIR / "meshes",
        Path(BOOSTER_ASSETS_DIR) / "robots" / "K1" / "meshes",
    ])
    missing = set()
    for mesh in root.iter("mesh"):
        name = Path(mesh.attrib["filename"]).name
        resolved = next((p / name for p in search_dirs if (p / name).is_file()), None)
        if resolved is None:
            missing.add(name)
        else:
            mesh.set("filename", resolved.resolve().as_posix())
    if missing:
        raise FileNotFoundError(
            "K1 meshes missing: " + ", ".join(sorted(missing))
            + ". Set SEO_K1_MESH_DIR to the complete INHA-RL/resources/K1/meshes directory."
        )

    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(xml).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "seo_walk_assets" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "K1_20dof.urdf"
    # Atomic publication also supports multiple training workers.
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".urdf", delete=False) as f:
            temporary = Path(f.name)
            f.write(xml)
        temporary.replace(target)
    return str(target)


def make_robot_cfg():
    cfg = copy.deepcopy(BOOSTER_K1_CFG)
    cfg.spawn.asset_path = prepare_urdf()
    cfg.init_state.joint_pos.update(K1_ARM_DEFAULT_POS)
    cfg.actuators["arms"] = BoosterDelayedPDActuatorCfg(
        joint_names_expr=K1_ARM_JOINT_NAMES,
        min_delay=0,
        max_delay=18,
        stiffness=20.0,  # Initial simulation values; not hardware-calibrated gains.
        damping=0.5,
        booster_joint_cfgs=BoosterJointR14(velocity_limit=18.0),
    )
    return cfg
