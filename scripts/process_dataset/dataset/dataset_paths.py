"""데이터셋별 경로 — 전처리 결과 루트와 물체 메시. 클립별 스크립트의 --dataset 이 이걸 쓴다.

표준 라이브러리만 쓴다 (env_pyroki 의 retarget_g1_pyroki.py 에서도 불러온다).

    sys.path.append(<저장소>/scripts/process_dataset/dataset)
    import dataset_paths
    dataset_paths.processed_root("grab")              # .../data/processed/grab
    dataset_paths.object_mesh("grab", "mug")          # 접촉·리타게팅에 쓰는 물체 메시 (물체 좌표계, m)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "source" / "robotis_sh5" / "data"
DATASETS = ("parahome", "grab", "omomo", "humoto")
SMPLX_MODEL_DIR = REPO_ROOT / "models_smplx_v1_1" / "models"   # 모든 데이터셋 공용 (GRAB 동봉 v1.0 과 FK 동일)


def _check(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"dataset 은 {DATASETS} 중 하나여야 합니다: {dataset!r}")


def processed_root(dataset: str) -> Path:
    _check(dataset)
    return DATA_DIR / "processed" / dataset


def object_mesh(dataset: str, obj: str) -> Path:
    """ParaHome 은 원본 스캔의 simplified/base.obj, GRAB·OMOMO 는 grab.py·omomo.py 가 원본을 옮겨 둔 .obj
    (OMOMO 는 스케일을 m 로 굽고 정점 중심으로 옮긴 것, 두 부품 물체는 손잡이). HUMOTO 는 전처리본에 들어 있는
    artist-made .obj (원본 데이터 없음)."""
    _check(dataset)
    if dataset == "parahome":
        return DATA_DIR / "raw" / "parahome" / "data" / "scan" / obj / "simplified" / "base.obj"
    return processed_root(dataset) / "assets" / "objects" / obj / "mesh" / f"{obj}.obj"


def contact_proxy_dir(dataset: str, obj: str) -> Path:
    """접촉 계산용 proxy (contact_proxy.py 가 만든다; HUMOTO 처럼 artist-made 메시인 데이터셋용)."""
    _check(dataset)
    return processed_root(dataset) / "assets" / "objects" / obj / "contact_proxy"
