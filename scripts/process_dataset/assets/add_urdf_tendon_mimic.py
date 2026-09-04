"""리타게팅용 URDF 의 손가락 J0 8개를 J1 의 mimic 관절로 표시합니다.

    /home/peunsu/anaconda3/envs/env_pyroki/bin/python \
        scripts/process_dataset/assets/add_urdf_tendon_mimic.py [--dry-run]

왜. 이 8개(robot0_{l,r}_{FF,MF,RF,LF}J0)는 액추에이터가 없고 PhysX 고정 텐던으로 J1 에 묶여
있습니다: q_J0 = (0.00805/0.00705) * q_J1 = 1.1418 * q_J1 (시뮬레이터 실측 기울기 0.913,
범위비 1.139). 즉 로봇의 실제 자유도는 URDF 가 말하는 73 이 아니라 65 입니다.

URDF 의 <mimic> 태그는 바로 이런 결합 관절을 표현하려고 있는 것이고, pyroki 가 정식으로
지원합니다(_robot_urdf_parser.py: mimic_multiplier / mimic_offset / mimic_act_indices,
actuated_indices 는 mimic 관절에 -1). FK 에서 value_multiplied = value_referenced *
mimic_multiplier 로 반영되므로:

  - solver 변수 차원이 73 -> 65 로 줄고
  - 결합이 구조적으로 정확히 만족되며(잔차 0, 가중치 없음)
  - 추가 비용 항이 필요 없습니다.

앞서 이것을 소프트 비용(tendon_couple, W_TENDON)으로 넣었는데, 정확한 항등식을 최소화 대상으로
바꾸는 것이라 부적절했습니다 — 절대 정확히 만족되지 않고, 다른 비용과 경쟁하며, 가중치가
튜닝 대상이 됩니다. 이 스크립트를 적용한 뒤 그 비용은 자동으로 비활성화됩니다(해당 관절이
actuated_names 에서 빠지므로 _tp 가 비고, W_TENDON 조건이 걸러냅니다).

이 URDF 는 리타게팅 전용입니다(env/시뮬레이터는 USD 를 씁니다). 멱등이고, 원본은
.pre_mimic.bak 로 백업합니다.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

_URDF = ("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1/"
         "urdf_pyroki/g1_shadow.urdf")
GEAR = 0.00805 / 0.00705          # 1.14184 — USD physxTendon gearing 비
PAIRS = [(f"robot0_{s}_{f}J1", f"robot0_{s}_{f}J0") for s in "lr"
         for f in ("FF", "MF", "RF", "LF")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=_URDF)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tree = ET.parse(args.urdf)
    root = tree.getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    todo = []
    for j1, j0 in PAIRS:
        el = joints.get(j0)
        if el is None:
            print(f"  [건너뜀] {j0}: URDF 에 없습니다")
            continue
        if el.find("mimic") is not None:
            m = el.find("mimic")
            print(f"  [멱등] {j0}: 이미 mimic (joint={m.get('joint')} mult={m.get('multiplier')})")
            continue
        if joints.get(j1) is None:
            print(f"  [건너뜀] {j0}: 구동 관절 {j1} 을 찾지 못했습니다")
            continue
        todo.append((j1, j0, el))

    print(f"\nmimic 으로 표시할 관절 {len(todo)}개  (multiplier = {GEAR:.5f})")
    for j1, j0, _e in todo:
        print(f"  {j0:<20} mimic-> {j1}")
    if args.dry_run or not todo:
        print("(dry-run — 아무것도 쓰지 않았습니다)" if args.dry_run else "할 일 없음")
        return

    bak = Path(args.urdf).with_suffix(".urdf.pre_mimic.bak")
    if not bak.exists():
        shutil.copy(args.urdf, bak)
        print(f"원본 백업 -> {bak.name}")

    for j1, _j0, el in todo:
        m = ET.SubElement(el, "mimic")
        m.set("joint", j1)
        m.set("multiplier", f"{GEAR:.6f}")
        m.set("offset", "0")
    ET.indent(tree, space="  ")
    tree.write(args.urdf, encoding="utf-8", xml_declaration=True)
    print(f"\nmimic 추가 완료: {len(todo)}개")


main()
