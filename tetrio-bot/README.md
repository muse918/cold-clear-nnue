# TETR.IO 배치 환경과 Cold Clear 기반 탐색기

Tetra League를 기본으로 하는 **C++17 배치 단위 환경·탐색기·JSONL 실행 파일**입니다.
기존 Cold Clear의 PPT 환경과 분리해 새로 구현했으며,
`experiments/spatial-nnue/kernels.cpp`의 native NNUE를 탐색 평가기에 연결했습니다.
기본 실행은 휴리스틱 평가를 사용합니다. 기존 `.snn` 모델을 지정하면 NNUE 또는
hybrid 평가와 증분 갱신을 사용할 수 있습니다.

## 구현 범위

| 영역 | 구현 |
|---|---|
| 보드 | 10×40, 하단부터 row mask, 숨겨진 20행, 일반/garbage/solid 출처, 줄 압축 |
| 배치 | 7종 미노, SRS/SRS+, 90°/180°, clutch, 기하학적 도달 가능성, spin provenance |
| spin | `none`, `T-spins`, `T-spins+`, `all`, `all+`, `all-mini`, `all-mini+`, `mini-only`, `handheld`, `stupid` |
| battle | 공식 combo/B2B, charge/surge, PC, downstack bonus, opener cancellation, 패킷 순서 |
| garbage | pending/active/spawn-ready, 상쇄, cap, receive/cancel multiplier, hole RNG, messiness, hardened/shielded |
| 상태 | current/hold/보이는 queue, hold lock, 이전 clear, frame, mutable multiplier/cap, 누적 sent |
| 탐색 | 세대별 beam과 DAG, 완전한 상태키, node/cache budget, deadline, 마지막 완료 depth, PV, 후보별 depth |
| NNUE | 부모 accumulator 재사용, 일반 배치 sparse update, 이벤트 refresh, 선택 가능한 row-seam update |
| 연결 | JSONL `state → search → apply`, C++ API, Triangle 변환 문서 |

조작 타이밍, gravity, DAS/ARR, lock delay, 렌더링, 계정 로그인, 네트워크 세션은
이 환경에 포함하지 않습니다. 반환되는 `path`는 도달 가능성을 증명하는 기하학적
경로입니다. 실제 프레임별 키 실행은 기존 봇 executor가 담당해야 합니다.
폭/높이 변경, 비표준 미노, bomb 모드는 지원하지 않습니다. 선택 가능한 추가 룰의
범위는 `info.rules`와 `Rules` 정의에 명시되어 있습니다.

## 빠른 실행

필요한 것은 C++17 compiler와 GNU Make입니다. JSON 의존성은
MIT 라이선스의 `nlohmann/json v3.12.0` header를 고정해 포함했습니다.
Python 3는 프로세스 smoke test, Node.js는 공식 소스 차등 검증에만 필요합니다.

레포 루트에서:

```sh
make -C tetrio-bot -j2
./tetrio-bot/build/tetrio-bot < tetrio-bot/examples/new_game.jsonl
```

출력은 입력 한 줄당 JSON 한 줄입니다. `search`는 보드를 변경하지 않습니다.
반환된 `action`을 `apply`에 보내면 로컬 시뮬레이션을 진행합니다.
실제 게임을 플레이하는 경우에는 executor의 lock 결과를 반영한 새 `state`를 보내세요.

```json
{"cmd":"configure","rules":{"preset":"tetra_league"},"search":{"max_depth":4,"beam_width":64,"max_nodes":12000,"time_budget_ms":50},"evaluation":{"mode":"heuristic"}}
{"cmd":"state","state":{"current":"T","hold":null,"queue":["I","O","S","Z","L"],"board":{"rows":[]}}}
{"cmd":"search","id":"move-1"}
```

위 상태에서 `queue`는 current를 제외한 **관측된 5개**입니다. 더 많은 미노를
전달하면 parser가 거부합니다. hold가 비어 있을 때 hold를 사용하면 그중 첫 미노를
배치하고, lock 이후 다음 current를 위해 한 개를 더 소비합니다. 관측 범위 끝에서는
unknown current로 멈추며 숨겨진 bag에서 뽑지 않습니다.

all-spin custom room 예시:

```json
{"cmd":"configure","rules":{"preset":"tetra_league","spin_mode":"all+","combo_mode":"multiplier","spin_180":true}}
```

`all`과 `all+`는 다른 공식 모드입니다. T의 corner 판정 실패 뒤 immobility fallback을
허용할지에 차이가 있습니다. 이름을 임의로 합치지 않았습니다.

### NNUE 실행

```sh
./tetrio-bot/build/tetrio-bot \
  --model experiments/spatial-nnue/results/final/seed918/attention.snn \
  --mode hybrid
```

모델은 한 번 로드됩니다. 실행 중에는 다음과 같이 정책을 변경할 수 있습니다.

```json
{"cmd":"configure","evaluation":{"mode":"hybrid","nnue_weight":0.25,"update":"refresh_events"}}
```

`update`는 `full`, `refresh_events`, `row_seams` 중 하나입니다. 현재 모델은 이전
surrogate 환경의 휴리스틱 label로 학습했습니다. 새 League battle outcome으로
재학습한 모델이 아니며, 아래 실험에서도 hybrid의 우위를 확인하지 못했습니다.
따라서 기본 baseline은 모델 없이 동작하는 휴리스틱 탐색입니다.

### C++ 사용

```cpp
#include "evaluator.hpp"
tetrio::Rules rules = tetrio::Rules::tetra_league();
tetrio::BotEvaluator evaluator; // evaluator must outlive searcher
tetrio::Searcher searcher(rules, evaluator);
tetrio::State state;
state.current = tetrio::Piece::T;
state.queue = {tetrio::Piece::I, tetrio::Piece::O};
const auto result = searcher.search(state);
if (result.action) {
    const auto next = tetrio::apply_action(state, *result.action, rules);
}
```

신뢰된 C++ `apply_action`은 generator/search가 생성한 action을 받는 함수입니다.
외부 JSON의 임의 좌표·spin은 `match_action`으로 생성된 합법 후보와 대조합니다.
`last_rotation`, kick index, path를 임의로 넣어 spin 공격을 만들 수 없습니다.

## 탐색 설계 결정

참고한 Cold Clear는
[`279edd7c3177ff8077f6a930193397814b281f27`](https://github.com/MinusKelvin/cold-clear/tree/279edd7c3177ff8077f6a930193397814b281f27)입니다.
원본은 가중 무작위 leaf selection, 세대 DAG, backpropagation, 미공개 bag에 대한
speculation을 사용합니다. 새 구현은 그중 상태 공유, hold/queue 처리, 상태 평가와
즉시 보상의 분리, root 재사용을 참고한 **결정론적 beam/DAG**입니다.
원본 알고리즘을 그대로 이식한 것은 아닙니다. [SEARCH.md](SEARCH.md)에 소스와
차이, budget·fallback·cache 계약을 정리했습니다.

기본값은 depth 4 / beam 64 / 12,000 processed edges / 50 ms입니다. deadline은
후보 사이에서 확인하므로 기하학적 배치 생성 하나와 child 평가 하나를 즉시 중단하지
않습니다. 50 ms의 엄격한 실시간 상한을 의미하지 않습니다. 시간이 부족하면 마지막으로
완료한 layer의 결과를 반환하고, 첫 layer도 완료하지 못하면 이미 찾은 합법 root 후보를
반환합니다. `time_budget_ms:0`은 시간 제한을 끄고 node budget으로 재현성 있게 비교합니다.

캐시 키에는 점유 보드뿐 아니라 garbage/solid 출처, hold/queue, combo/B2B,
frame, dynamic rates, incoming 패킷, RNG 등 미래 전이에 영향을 주는 상태가 들어갑니다.
해시값만 비교하지 않고 전체 canonical key를 비교합니다. 새 preview가 보이면 상태가
달라지므로 보수적으로 그래프를 다시 만듭니다. 아래 실험의 cross-move reuse는 0이고,
한 탐색 안의 DAG 공유는 실제로 발생했습니다. 향후 queue와 무관한 board accumulator
cache를 분리하는 것은 별도의 최적화 과제입니다.

## 행 삭제와 증분 평가

`BoardDelta`는 배치한 4칸, 삭제 mask, `child row → parent row` 매핑,
실제로 삽입한 garbage 행 수를 전달합니다. 일반 배치는 바뀐 cell 및 주변 local token의
기여만 갱신합니다. 삭제·garbage 이벤트에서는 기본적으로 전체 accumulator를 재계산하며,
`row_seams`로 살아남은 행 주변의 경계 변화만 계산하는 기존 kernel도 선택할 수 있습니다.

행 삭제가 여러 칸을 한꺼번에 이동시키므로 absolute position feature에서 삭제한 행의
기여만 빼는 방식은 정확하지 않습니다. 행 매핑을 전달해 이동을 반영하거나 해당 이벤트를
refresh해야 합니다. 또한 clear 빈도는 정책과 상황에 따라 달라지므로 이벤트가 드물다는
가정만으로 성능 우위를 선언하지 않습니다. 이번 작업은 실제 환경의 전이에 대한 정확성까지
검증했으며, kernel 자체의 timing 비교는 이전 [spatial NNUE 실험](../experiments/spatial-nnue/README.md)에 있습니다.

입력 head의 미노는 child의 `state.current`입니다. `queue.front()`를 넣으면 한 미노
앞선 정보를 잘못 사용하는 문제가 생깁니다. unknown horizon에서는 명시적으로 board
heuristic을 사용합니다. 콤보·B2B·pending 위험 및 attack 보상은 현재 손으로 지정한
값이며 모델이 battle 문맥을 이미 학습했다고 가정하지 않습니다.

## 검증 결과

규칙 기준은 사용자가 제공한 **TETR.IO Beta 1.7.8 / engine 19**입니다. SHA-256:
`6b682a1226dd2ae66ce19a08e944df92e7f175bccaa48cca7269bcd78f39a27a`.
원본 클라이언트 전체는 저장소에 포함하지 않습니다. Oracle script는 사용자가 지정한
파일의 hash를 확인하고 실제 함수들을 런타임에 추출해 실행합니다.

| 검증 | 기록된 결과 |
|---|---|
| 공식 attack/cancellation/garbage/enqueue 차등 검사 | 16,592 cases, 0 mismatches |
| 공식 geometry/kicks/spin/clutch/placement-set 차등 검사 | 28,286 cases, 0 mismatches |
| 반환 경로 replay | 공식 primitive로 2,245개, native regression으로 4,864개 |
| battle 회귀 및 숫자 범위 | 80 checks; UBSan/float-cast-overflow 검사도 통과 |
| 탐색 | 독립적 2·3-ply exhaustive 비교 등 13 groups 통과 |
| JSON schema·상태 보존·spin 검증 | 35 checks 통과 |
| 실제 전이의 NNUE 증분/전체 비교 | 11,469 edges, accumulator 성분 6,239,136개 비교 |
| NNUE 최대 accumulator 절대 오차 | RefreshEvents 1.91e-5, RowSeams 3.05e-5 |
| 프로세스 통합 | 105 JSONL requests, 24 legal locks, 10 lines, 잘못된 요청 뒤 복구 |
| 오프라인 게임 | 24 runs × 120 locks, 합법성·queue·hold·frame·garbage 보존 검사 통과 |

geometry 검증은 문서화된 정수 배치 그래프에 대한 것입니다. 공식 falling-y가 분수일 때
buffer 최상단의 충돌 판정이 달라지는 부분은 보수적인 canonical phase를 사용합니다.
모든 gravity phase와 lock timing의 동등성을 뜻하지 않습니다.
[geometry audit](tests/geometry/README.md), [rules audit](tests/rules/RULES_AUDIT.md),
[원시 gameplay 결과](tests/gameplay_report.json)를 참고하세요.

### 실제 환경의 오프라인 탐색 실험

정책마다 같은 8개 7-bag queue, 최대 120 lock, 5 preview를 사용했습니다. 4개는 clean,
4개는 12 lock마다 고정 hole의 ready garbage 2줄을 받았습니다. 시간은 외부에서
30 frames/lock으로 공급했습니다. 두 beam 정책은 depth 3 / beam 16 /
2,000 edges / deadline 없음으로 실행했습니다.

| 평가·탐색 | 배치 | 삭제 | 보낸 공격 | 상쇄 | 받은 garbage | 평균 search ms |
|---|---:|---:|---:|---:|---:|---:|
| 1수 휴리스틱 | 960 | 431 | 90 | 4 | 73 | 0.333 |
| 3수 beam, 휴리스틱 | 960 | 430 | 202 | 11 | 64 | 11.364 |
| 3수 beam, 25% NNUE hybrid | 960 | 425 | 183 | 16 | 62 | 12.921 |

24개 실행 모두 살아서 120-lock cap에 도달했습니다. 두 beam 정책 모두 960개 결정 중
949개에서 depth 3을 완료했고, 11개는 budget 때문에 완료된 depth 2로 반환했습니다.
휴리스틱 beam의 within-search cache hit는 108,368개, hybrid는 109,920개였습니다.

이는 상대가 없는 작은 기능 실험입니다. 대전 승률, 장기 생존, 실제 입력 실행을
검증한 결과가 아닙니다. shared-host timing에는 기하학적 생성과 전이 전체 비용이
포함되어 NNUE kernel 속도 비교에도 사용할 수 없습니다. 현재 hybrid는 보낸 공격량이
휴리스틱보다 낮으므로, 기존 가중치를 바로 주력 평가기로 승격할 근거가 없습니다.

## 테스트 재현

```sh
make -C tetrio-bot -j2 test
make -C tetrio-bot verify-official TETRIO_JS=/absolute/path/to/tetrio.js
make -C tetrio-bot gameplay
```

`gameplay`는 `tests/gameplay_report.json`을 해당 머신의 새 결과로 덮어씁니다.
정확한 공식 부동소수점 연산 순서가 필요한 코드이므로 `-ffast-math`를 사용하지 마세요.
이 실행 환경처럼 `/tmp`가 없는 경우에는 먼저 `tetrio-bot/build`를 만들고
`TMPDIR`을 그 절대 경로로 지정해 Make를 실행하면 됩니다.

## 실제 봇에 연결할 때 필요한 경계

[ADAPTER.md](ADAPTER.md)는 pinned Triangle의 실제 필드를 기준으로 작성했습니다.
Triangle board는 이미 bottom-first입니다. `queue.raw()` 전체를 프리뷰로 전달하면
안 됩니다. stock compact wrapper state는 pending 확인 상태·hold lock·RNG 등의
정보를 잃으므로 확장된 snapshot이 필요합니다.

기본 League는 `all-mini+`, B2B charging, charge base 0, PC garbage 5를 사용합니다.
공식 PC difficult clear는 기본 설정에서 B2B가 2 증가할 수 있습니다. Triangle의
boolean/offset 표현만 그대로 복사하면 이 의미를 보존하지 못합니다. 또한 pinned
Triangle의 config `garbageCap`은 multiplier getter를 읽는 오류가 있으므로 실제 cap
getter를 사용해야 합니다. 소스 위치와 변환 계약은 위 문서에 연결했습니다.

네트워크 ACK/passthrough reconciliation을 완료한 상태만 전달하세요. `active`와
`spawn_ready`는 caller가 알려 주는 실제 상태이며 `advance`만으로 자동 활성화하지
않습니다. 늦은 snapshot에는 현재 mutable multiplier/cap을 전달해야 합니다.
이번 결과물은 이 경계를 가진 native search/environment이며, 봇 계정의 온라인
플레이나 Triangle의 frame-key executor를 실행한 결과물은 아닙니다.

## 소스와 라이선스

새 native 소스는 레포 루트의 [MPL-2.0](../LICENSE)을 따릅니다.
`third_party/nlohmann/json.hpp`는 `v3.12.0`의 원본 header이며 별도
[MIT notice](third_party/NLOHMANN_LICENSE.MIT)를 포함했습니다. 해당 upstream blob은
`82d69f7c5d044c9887c96b90c97f5639083ecd14`로 확인했습니다.
공식 `tetrio.js` 전체와 계정 자격 증명은 배포물에 포함하지 않습니다.
