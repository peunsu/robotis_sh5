# 저장소 루트의 local_paths.env 에서 기계마다 다른 경로를 읽는다 (git 에 올리지 않는 파일).
# 이미 환경변수로 준 값이 이긴다 (예: PY=... bash train_....sh). 파이썬 쪽은 scripts/local_paths.py.
#
#   source "<저장소>/scripts/local_paths.sh"
#   require_local_path PY PY_PYROKI      # 없으면 어디에 쓰라는 메시지와 함께 종료

_LOCAL_PATHS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/local_paths.env"

if [[ -f "${_LOCAL_PATHS_FILE}" ]]; then
    while IFS='=' read -r _lp_key _lp_value; do
        [[ -n "${!_lp_key:-}" ]] && continue
        _lp_value="${_lp_value%\"}"; _lp_value="${_lp_value#\"}"
        _lp_value="${_lp_value%\'}"; _lp_value="${_lp_value#\'}"
        export "${_lp_key}=${_lp_value}"
    done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${_LOCAL_PATHS_FILE}")
fi

require_local_path() {
    local _lp_key
    for _lp_key in "$@"; do
        if [[ -z "${!_lp_key:-}" ]]; then
            echo "ERROR: ${_lp_key} 가 설정되지 않았습니다 — ${_LOCAL_PATHS_FILE} 에 ${_lp_key}=<절대 경로> 를 쓰세요" \
                 "(예시: local_paths.env.example)" >&2
            exit 1
        fi
    done
}
