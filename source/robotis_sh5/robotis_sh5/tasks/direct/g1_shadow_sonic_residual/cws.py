"""접촉을 "물체를 어떻게 움직일 수 있는가"로 바꾸는 계산 (CHORD, arXiv 2607.00033).

용어를 먼저 풀어 씁니다.

- 접촉 하나가 물체에 주는 효과는 두 가지입니다: 미는 힘(3방향)과 그 힘이 물체 중심에 대해
  만드는 회전 효과(3축). 합쳐서 숫자 6개이고, 논문은 이 묶음을 렌치(wrench)라고 부릅니다.
- 손가락은 표면에 수직으로만 미는 게 아니라 마찰 덕분에 비스듬히도 밉니다. 밀 수 있는 방향의
  범위가 원뿔이 됩니다(마찰 원뿔). 반각은 atan(마찰계수)입니다.
- 손 전체의 능력은 미리 정해둔 방향 512개 각각에 대해 "이 방향으로 최대 얼마나 밀 수 있나"를
  기록한 목록으로 요약합니다. 논문 용어로는 지지함수(support function)입니다.

기본 경로(support)는 논문 방식 그대로 원뿔을 대표 방향으로 근사합니다. 재현이 목적이기 때문입니다.
support_exact는 원뿔 위 최대값의 닫힌 형태로, 근사가 맞게 구현됐는지 대조하는 용도로 둡니다
(대표 방향을 128개로 늘리면 0.01% 이내로 일치).

논문 공개 구현(nvidia-isaac/video_to_data, robotic_grounding)과 대조한 결과입니다 (2026-09-02).

같은 것: 방향 512개, 기저는 균일 S^5, 회전 성분을 물체 크기로 나누기, 지지함수는 생성자에 대한
max 후 0 이상으로 자르기. 회전 성분 나누기는 이전 주석에 "논문에는 이 처리가 없다"고 적혀
있었는데 사실이 아닙니다 — utils.py:compute_wrench_space 와 utils_jit.py:wrench_support_one_body_jit
가 cat((forces, torques / rc)) 로 같은 처리를 합니다. rc 는 tracking_command.py:807 에서 물체
body 별 object_mesh_radius(기본 0.05 m)로 넘어갑니다. 기저 쪽 sample_wrench_space_basis_scaled
는 이름과 달리 tracking_command.py:713 에서 rc=1.0 으로 호출되어 무연산입니다.

물체 크기(rc)와 모멘트 팔은 이후 논문에 맞췄습니다 (env.py 의 cws-rc-mesh / cws-com 마커).
rc 는 메시 정점 중심에서 최대 거리(칼 13.11 cm), 모멘트 팔은 물리 COM 기준입니다. 이전에는
접촉점 노름의 0.9 분위(칼 5.60 cm)와 body 원점을 썼는데, rc 가 2.34 배 작아 토크에 그만큼
민감했습니다.

남은 차이 하나.

- 접촉 판정. 논문은 힘 임계가 없습니다. 활성 판정이 접촉점 위치의 노름(> 1e-3)뿐이고 이는
  "PhysX 가 접촉점을 보고했는가"와 같습니다. 논문 ContactSensorCfg 의 force_threshold=0.1 은
  렌치와 무관합니다 — IsaacLab 에서 그 값은 track_air_time 의 체공/접촉 시간과 시각화 마커에만
  쓰입니다(contact_sensor.py:422, 519). 우리는 cws_force_thresh(0.1 N)를 추가로 요구합니다.

센서 배치는 논문이 물체측(센서 1개 + 손 링크 32개 필터), 우리가 손측(센서 32개 + 물체 1개
필터)인데 출력이 동등합니다. force_matrix_w / contact_pos_w 의 형상이 (센서, body, 필터, 3)
이고 contact_pos_w 가 "쌍 사이 접촉점들의 평균"이라, 같은 (손링크, 물체) 쌍에 대해 양쪽이 같은
값 하나를 냅니다. 힘은 뉴턴 제3법칙으로 부호만 반대이고 우리가 뒤집습니다. 논문의
history_length=3 도 [:, 0](최신)만 읽으므로 우리 history_length=1 과 같은 값입니다.

논문의 body 차원(N,B,K)의 B 는 물체 body 수(num_bodies = object_position_e.shape[1])이지 손
링크가 아닙니다. 강체 단일 물체만 스폰하는 우리 환경에서는 B=1 이므로, 링크 전체에 대해 max 를
취하는 우리 구조가 논문과 같습니다.
"""

import torch


def make_basis(n_dir: int = 512, seed: int = 0, device="cpu") -> torch.Tensor:
    """비교에 쓸 방향 n_dir개. (6, n_dir).

    사람 쪽과 로봇 쪽이 반드시 같은 방향 집합을 써야 하므로 시드를 고정합니다.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    b = torch.randn(6, n_dir, generator=g)
    return (b / b.norm(dim=0, keepdim=True)).to(device)


def support_exact(basis, points, normals, active, mu: float, length_scale: float) -> torch.Tensor:
    """방향마다 "최대 얼마나 밀 수 있나". -> (..., n_dir).

    basis (6, n_dir), points/normals (..., L, 3), active (..., L).
    points와 normals는 물체 기준 좌표여야 합니다(물체가 움직여도 값이 안 변하도록).
    normals는 접촉면이 물체를 미는 방향입니다.

    유도. 원뿔 안의 단위 힘 f가 방향 b=(b_f, b_m)에 기여하는 값은

        b·w(f) = b_f·f + b_m·(p×f)/L = f·g,    g = b_f + (b_m×p)/L

    (스칼라 삼중곱 b_m·(p×f) = f·(b_m×p)). 즉 f·g를 원뿔 안에서 최대화하는 문제이고, g가 원뿔
    안이면 |g|, 밖이면 |g|cos(θ-α)입니다(θ = g와 법선 사이 각, α = 원뿔 반각). 접촉이 없거나
    최대값이 음수인 경우는 "안 민다"는 선택지가 있으므로 0입니다.

    g를 직접 만들면 (..., L, n_dir, 3) 크기의 중간 텐서가 생겨 환경 수천 개에서 메모리를
    잡아먹으므로, 필요한 스칼라량만 삼중곱으로 풀어 (..., L, n_dir) 행렬곱 네 번으로 구합니다.
    """
    bf, bm = basis[:3].transpose(0, 1), basis[3:].transpose(0, 1)      # (n_dir,3)
    Ls = float(length_scale)
    n = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    pxn = torch.cross(points, n, dim=-1)                               # (...,L,3)
    bfxbm = torch.cross(bf, bm, dim=-1)                                # (n_dir,3)
    # g·n = b_f·n + b_m·(p×n)/L        ((b_m×p)·n = b_m·(p×n))
    gn = n @ bf.transpose(0, 1) + (pxn @ bm.transpose(0, 1)) / Ls
    # |g|² = |b_f|² + 2 p·(b_f×b_m)/L + (|b_m|²|p|² - (b_m·p)²)/L²
    bm_p = points @ bm.transpose(0, 1)                                 # (...,L,n_dir)
    g2 = (bf.pow(2).sum(-1)                                            # (n_dir,)
          + 2.0 * (points @ bfxbm.transpose(0, 1)) / Ls
          + (bm.pow(2).sum(-1) * points.pow(2).sum(-1, keepdim=True) - bm_p.pow(2)) / (Ls * Ls))
    gnorm = g2.clamp(min=0.0).sqrt()

    ca = 1.0 / (1.0 + mu * mu) ** 0.5                                  # cos(원뿔 반각)
    sa = mu * ca                                                       # sin(원뿔 반각)
    cos_t = (gn / gnorm.clamp(min=1e-9)).clamp(-1.0, 1.0)              # g와 법선 사이 각의 cos
    sin_t = (1.0 - cos_t * cos_t).clamp(min=0.0).sqrt()
    # g가 원뿔 안이면 |g|, 밖이면 |g|cos(θ-α)
    val = gnorm * torch.where(cos_t >= ca, torch.ones_like(cos_t), cos_t * ca + sin_t * sa)
    val = val.clamp(min=0.0) * active.unsqueeze(-1).to(val.dtype)
    return val.max(dim=-2).values                                      # 링크에 대한 최대 -> (...,n_dir)


def cws_reward(sigma_h, sigma_r, beta: float, v: float, active_eps: float = 1e-3) -> torch.Tensor:
    """사람 목록과 로봇 목록을 비교해 0~1 점수로. 둘 다 (..., n_dir).

    논문 utils_jit.py:contact_wrench_support_reward_jit 와 같은 집계입니다.

    로봇이 사람의 (1-beta)배보다 못 밀면 부족, (1+beta)배를 넘겨 밀면 과잉으로 보고, 그 사이
    여유 범위 안에서는 두 항이 모두 0입니다. 여기까지는 이전과 같고, 합치는 방식이 다릅니다.

    - 방향마다 따로 exp(-부족²+과잉²/v)를 걸어 부분 점수를 냅니다.
    - 분자는 사람도 요구하고(cmd) 로봇도 실제로 내는(cur) 방향만 셉니다. 사람이 요구하지만
      로봇이 못 내는 방향은 0으로 들어가 분모에만 남으므로 비례 감점이 됩니다.
    - 분모는 사람이 요구하는 방향 수. 아무 방향도 요구하지 않으면 0점입니다.

    이전 구현은 512개 방향의 부족분을 SUM 한 뒤 단일 exp 를 걸었습니다(exp(-sum/v)). Jensen
    부등식으로 그 형태가 방향별 평균보다 훨씬 작아, 512개 방향을 거의 모두 재현해야 0이 아닌
    값이 나왔습니다. 실측상 96%가 정확히 0.000 이어서 실패들 사이를 구분하지 못했습니다.
    논문 형태는 방향 하나만 맞아도 1/cmd_num 만큼 점수가 나옵니다.

    active_eps 는 논문의 supports > 1e-3 판정과 같습니다.
    """
    cmd_active = sigma_h > active_eps
    cur_active = sigma_r > active_eps
    better = ((1.0 - beta) * sigma_h - sigma_r).clamp(min=0.0)      # 부족
    too_large = (sigma_r - (1.0 + beta) * sigma_h).clamp(min=0.0)   # 과잉
    loss = better.pow(2) + too_large.pow(2)                         # (...,n_dir) 방향별
    both = (cmd_active & cur_active).to(loss.dtype)
    cmd_num = cmd_active.sum(-1).to(loss.dtype).clamp(min=1e-6)
    return (both * torch.exp(-loss / v)).sum(-1) / cmd_num


# ── [ROLLBACK MARKER: cws-diag] 보정이 필요 없는 진단 지표 (2026-08-18) ─────────────────────
# cws_reward 가 합산 후 단일 exp 였던 시절에는 점수가 구조적으로 항상 0이었습니다(실측: 96%가
# 정확히 0.000). 그래서 v 와 무관한 진단 지표가 필요했습니다. 지금은 cws_reward 가 논문의
# 방향별 평균 형태라 그 자체로 실패들을 구분하지만, 아래 둘은 여전히 v 와 무관하고 방향 수에도
# 스케일되지 않아 v 를 보정할 때의 기준값으로 유용합니다.
# cws_coverage 는 논문의 force_closure_reward / missed_contact_penalty 와 대응합니다(우리는
# 보상이 아니라 진단으로만 씁니다).
def cws_coverage(sigma_h, sigma_r, beta: float, eps: float = 1e-4):
    """레퍼런스가 실제로 요구하는 방향 중, 로봇이 (1-beta)배 이상 낼 수 있는 방향의 비율 ∈[0,1].

    sigma_h 가 0 인 방향(레퍼런스도 그쪽으로는 못 미는 방향)은 분모에서 제외합니다 —
    포함하면 "요구가 없어서 만족"이 커버리지를 부풀립니다.
    """
    need = sigma_h > eps                                               # (...,n_dir)
    ok = (sigma_r >= (1.0 - beta) * sigma_h) & need
    return ok.sum(-1).float() / need.sum(-1).clamp(min=1).float()


def cws_deficit(sigma_h, sigma_r, beta: float):
    """방향당 평균 부족분 (sigma_h 단위). 0 이면 부족 없음. cws_v 를 보정할 때의 기준값."""
    need = sigma_h > 1e-4
    lo = ((1.0 - beta) * sigma_h - sigma_r).clamp(min=0.0) * need
    return lo.sum(-1) / (sigma_h * need).sum(-1).clamp(min=1e-6)


# ---------------------------------------------------------------------------------------------
# 논문 방식(원뿔을 대표 방향으로 근사). 재현이 목적이므로 이쪽이 기본 경로입니다.

def friction_cone(normal, mu: float, n_edge: int = 4):
    """마찰 원뿔의 대표 방향들. normal (...,3) -> (...,n_edge+1,3), 각각 크기 1.

    논문(utils_jit.py:friction_cone_edges_jit)처럼 원뿔 옆면 n_edge 개에 순수 법선 1개를 더해
    돌려줍니다. 옆면만 쓰면 기저 방향이 원뿔 "안쪽"에 떨어질 때 최대값을 놓칩니다 — 옆면은 모두
    법선에서 atan(mu) 만큼 기울어져 있어서, 축 방향으로는 cos(atan(mu)) 배만 낼 수 있는 것으로
    과소평가됩니다. support_exact 는 원뿔 내부를 포함한 닫힌 형태라 이 경우도 맞게 처리하므로,
    법선 추가는 support 를 support_exact 쪽으로 붙입니다.
    """
    n = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ref = torch.zeros_like(n)
    ref[..., 2] = 1.0
    alt = torch.zeros_like(n)
    alt[..., 0] = 1.0
    ref = torch.where((n[..., 2:3].abs() > 0.9).expand_as(ref), alt, ref)
    t1 = torch.cross(n, ref, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    t2 = torch.cross(n, t1, dim=-1)
    phi = torch.arange(n_edge, device=n.device, dtype=n.dtype) * (2 * torch.pi / n_edge)
    tang = torch.cos(phi)[:, None] * t1.unsqueeze(-2) + torch.sin(phi)[:, None] * t2.unsqueeze(-2)
    e = n.unsqueeze(-2) + mu * tang
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.cat([e, n.unsqueeze(-2)], dim=-2)                 # 옆면 + 순수 법선


def support(basis, points, normals, active, mu: float, length_scale: float, n_edge: int = 16,
            link_chunk: int = 0):
    """논문 방식: 원뿔을 대표 방향 n_edge개(+법선 1개)로 근사하고 그중 최대. -> (..., n_dir).

    n_edge 는 논문 공개 구현의 num_friction_cone_edges = 8 과 맞췄습니다. 대표 방향을 뽑으려면 고정 기준축이 필요한데
    그래서 물체를 회전시키면 값이 달라집니다 — 물리가 아니라 근사 때문에 생기는 오차이고, 실측
    상대오차(중앙/최대)는 4개 5.1%/42%, 8개 1.1%/10.2%, 16개 0.3%/2.5%, 32개 0.08%/0.6%입니다.
    우리 물체는 클립 내내 자세가 바뀌므로 같은 파지가 프레임마다 다른 점수를 받게 됩니다. 다만
    그 실측은 mu=1.0(반각 45도) 에 법선 생성자가 없던 시절 값입니다. 현재 설정(mu=0.1, n_edge=8,
    법선 추가)에서 support_exact 대비 상대오차는 중앙 0.147%, 최대 절대오차 0.026 입니다(sigma
    최대 3.73 의 0.7%). 상대오차 최대치는 100% 로 보이지만 support_exact 가 거의 0인 띠에서만
    나오고(>50% 오차 셀 0.031%, 그 지점 최대 0.018), 절대 크기로는 무의미합니다. 법선 생성자
    추가는 원뿔이 넓을 때 특히 효과가 커서 mu=1.0/n_edge=16 에서 최대 절대오차를 0.967 -> 0.286
    으로 줄입니다. 중간 배열이 (환경 x n_dir x 링크*(n_edge+1))로 커지는 것이 비용입니다.
    """
    L = points.shape[-2]
    chunk = int(link_chunk) if link_chunk else L
    Bt = basis.transpose(0, 1)                                         # (n_dir,6)
    out = None
    # 링크를 나눠 처리하고 최대만 누적합니다. 한 번에 하면 중간 배열이
    # (환경 x n_dir x 링크*n_edge)라, 환경 2048/방향 512/링크 32/대표방향 16에서 2 GB가 넘습니다.
    # 최대는 결합법칙이 성립하므로 나눠서 구해도 결과가 같습니다.
    for i in range(0, L, chunk):
        pc, nc, ac = points[..., i:i + chunk, :], normals[..., i:i + chunk, :], active[..., i:i + chunk]
        e = friction_cone(nc, mu, n_edge)
        pe = pc.unsqueeze(-2).expand_as(e)
        w = torch.cat([e, torch.cross(pe, e, dim=-1) / length_scale], dim=-1)
        w = w * ac.unsqueeze(-1).unsqueeze(-1).to(w.dtype)
        w = w.flatten(-3, -2).transpose(-1, -2)                        # (...,6,chunk*n_edge)
        v = (Bt @ w).max(dim=-1).values                                # (...,n_dir)
        out = v if out is None else torch.maximum(out, v)
    return out.clamp(min=0.0)
