#!/usr/bin/env python3
# Sprint scene builder: dual prefixed SO-101 + dinner-table objects -> assets/scene/dinner_table_scene.xml
# Source model: TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.xml (new calibration).
# Prefixing avoids MuJoCo duplicate-name errors when including the arm twice.
"""Usage: python scripts/build_scene.py [--out assets/scene/dinner_table_scene.xml]"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO_ROOT, "assets", "robots", "so101", "arm_template.xml")

TABLE_Z = 0.40
ARM_A_POS = (-0.20, 0.0, TABLE_Z + 0.02)
ARM_B_POS = (0.20, 0.0, TABLE_Z + 0.02)

NAME_ATTRS = ("name", "joint", "body", "geom", "site", "camera", "tendon", "material", "mesh")


def prefix_tree(elem, prefix, skip_tags=("asset",)):
    for el in elem.iter():
        if el.tag in skip_tags:
            continue
        for attr in NAME_ATTRS:
            if attr in el.attrib and attr != "material" and attr != "mesh":
                el.attrib[attr] = prefix + el.attrib[attr]


def build_arm(prefix, pos):
    tree = ET.parse(TEMPLATE)
    root = tree.getroot()
    world = root.find("worldbody")
    arm_body = world.find("body")  # single top-level 'base' body
    arm = copy.deepcopy(arm_body)
    arm.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    if prefix == "b_":
        # Mirror arm B to face the shared workspace (180 deg yaw about z).
        arm.set("quat", "0 0 0 1")
    # Collision masking: ALL arm meshes off. Rationale (sprint log): the live jaw
    # meshes clip props during transit (bottle ejected at ~0.4 m/s on some seeds).
    # All grasps are coupling-zone based (disclosed oracle-only flag), so no grasp
    # needs mesh contact; props still collide with table/floor/drawer/each other.
    for g in arm.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
    prefix_tree(arm, prefix)
    # Wrist camera attached to the (prefixed) wrist body for that arm.
    wrist = None
    for b in arm.iter("body"):
        if b.get("name") == prefix + "wrist":
            wrist = b
    if wrist is not None:
        cam = ET.SubElement(wrist, "camera")
        cam.set("name", prefix + "wrist_cam")
        cam.set("pos", "0 -0.02 0.05")
        cam.set("xyaxes", "1 0 0 0 1 0")
    # Actuators referencing this arm's joints.
    acts = []
    for act in root.find("actuator"):
        if act.get("joint", "").split("_")[0] in ("shoulder", "elbow", "wrist", "gripper"):
            a = copy.deepcopy(act)
            a.set("name", prefix + a.get("name"))
            a.set("joint", prefix + a.get("joint"))
            acts.append(a)
    return arm, acts


def table_objects():
    wb = ET.Element("worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 3.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "2 2 0.1",
                               "pos": "0 0 0", "material": "groundplane"})
    ET.SubElement(wb, "geom", {"name": "tabletop", "type": "box", "size": "0.30 0.22 0.015",
                               "pos": f"0 0 {TABLE_Z}", "rgba": "0.55 0.38 0.22 1"})
    # Drawer: sliding box on the table, handle site for the oracle.
    drawer = ET.SubElement(wb, "body", {"name": "drawer", "pos": f"-0.10 0 {TABLE_Z + 0.042}"})
    ET.SubElement(drawer, "joint", {"name": "drawer_slide", "type": "slide", "axis": "1 0 0",
                                    "range": "0 0.12", "damping": "0.5"})
    ET.SubElement(drawer, "geom", {"name": "drawer_box", "type": "box", "size": "0.06 0.05 0.025",
                                   "rgba": "0.7 0.55 0.35 1"})
    ET.SubElement(drawer, "site", {"name": "drawer_handle", "pos": "-0.065 0 0.01"})
    ET.SubElement(drawer, "inertial", {"pos": "0 0 0", "mass": "0.15",
                                       "diaginertia": "0.0002 0.0002 0.0002"})

    def free_prop(parent, name, ptype, size, pos, rgba, mass="0.05"):
        b = ET.SubElement(parent, "body", {"name": name, "pos": pos})
        ET.SubElement(b, "freejoint")
        ET.SubElement(b, "geom", {"name": name + "_geom", "type": ptype, "size": size, "rgba": rgba})
        ET.SubElement(b, "inertial", {"pos": "0 0 0", "mass": mass,
                                      "diaginertia": "0.00005 0.00005 0.00005"})
        return b

    tz = TABLE_Z + 0.015  # tabletop surface plane
    # Spawn centers = surface + half-height + 2mm (zero penetration at t0).
    free_prop(wb, "fork", "box", "0.008 0.008 0.04", f"-0.02 0.10 {tz + 0.042}", "0.8 0.8 0.85 1", "0.02")
    free_prop(wb, "plate", "cylinder", "0.05 0.05 0.008", f"0.06 -0.06 {tz + 0.010}", "0.95 0.95 0.97 1", "0.12")
    mug = free_prop(wb, "mug", "cylinder", "0.035 0.035 0.05", f"0.10 0.08 {tz + 0.052}", "0.8 0.2 0.2 1", "0.10")
    ET.SubElement(mug, "site", {"name": "mug_rim_site", "pos": "0 0 0.055"})
    free_prop(wb, "bottle", "cylinder", "0.04 0.04 0.08", f"-0.06 -0.14 {tz + 0.082}", "0.2 0.5 0.8 1", "0.25")
    # Fixed task sites.
    ET.SubElement(wb, "site", {"name": "plate_target_site", "pos": "0.06 -0.06 0.425"})
    ET.SubElement(wb, "site", {"name": "handoff_zone", "pos": "0 0 0.55"})
    # Overhead camera.
    ET.SubElement(wb, "camera", {"name": "overhead", "pos": "0 -0.75 0.95", "xyaxes": "1 0 0 0 0.35 0.94"})
    return wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "assets", "scene", "dinner_table_scene.xml"))
    args = ap.parse_args()

    tpl = ET.parse(TEMPLATE).getroot()
    out = ET.Element("mujoco", {"model": "dinner_bimanual_so101"})
    ET.SubElement(out, "compiler", {"angle": "radian", "meshdir": "../robots/so101", "autolimits": "true"})
    ET.SubElement(out, "option", {"integrator": "implicitfast", "timestep": "0.002", "gravity": "0 0 -9.81"})
    for default in tpl.findall("default"):
        out.append(copy.deepcopy(default))
    # Shared mesh/material assets from the vendored arm (single copy, shared by both arms).
    out.append(copy.deepcopy(tpl.find("asset")))
    # Table textures/materials.
    asset = out.find("asset")
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker",
                                     "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
                                     "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane",
                                      "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})
    vis = ET.SubElement(out, "visual")
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "960"})
    world = ET.SubElement(out, "worldbody")
    # Table + objects first (fixed sites/cameras).
    for child in list(table_objects()):
        world.append(child)
    # Two prefixed arms.
    act_root = ET.SubElement(out, "actuator")
    drawer_act = ET.SubElement(act_root, "position", {"name": "drawer", "joint": "drawer_slide",
                                                      "kp": "500", "forcerange": "-8 8", "ctrlrange": "0 0.12"})
    for prefix, pos in (("a_", ARM_A_POS), ("b_", ARM_B_POS)):
        arm, acts = build_arm(prefix, pos)
        world.append(arm)
        for a in acts:
            act_root.append(a)
    ET.SubElement(out, "equality")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ET.indent(out, space="  ")
    ET.ElementTree(out).write(args.out, xml_declaration=True, encoding="unicode")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
