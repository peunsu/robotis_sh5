"""기계마다 다른 절대 경로를 저장소 루트의 local_paths.env 에서 읽는다 (git 에 올리지 않는 파일).

읽는 순서: 같은 이름의 환경변수 → local_paths.env → 오류. 셸 스크립트는 scripts/local_paths.sh 로
같은 파일을 읽는다. 표준 라이브러리만 쓴다 (env_pyroki 에서도 불러온다).

    sys.path.append(<저장소>/scripts)
    import local_paths
    local_paths.get("GR00T_ROOT")
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "local_paths.env"


def _read_file() -> dict[str, str]:
    out: dict[str, str] = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip("\"'")
    return out


def get(key: str) -> str:
    value = os.environ.get(key) or _read_file().get(key)
    if not value:
        raise RuntimeError(f"{key} 가 설정되지 않았습니다 — {ENV_FILE} 에 {key}=<절대 경로> 를 쓰세요 "
                           f"(예시: {ENV_FILE.name}.example)")
    return value
