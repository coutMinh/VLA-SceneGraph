# Study guide — đọc nhanh để viết report (Part 1 + ý tưởng Part 4)

Mục tiêu: trong ~1,5 ngày, đọc **đủ để tự viết** Part 1 và Part 4, không đọc hết 14 bài.
Ghi chú chi tiết đã có (chỉ mở được trên máy, nằm ngoài repo này):
`../study/literature_matrix.md` (14 nguồn, CSV là bản gốc),
`../study/scene_graph_synthesis.md` (so sánh theo trục),
`../study/diffusion_vs_flow_matching.md` (công thức + bảng so sánh).
Các ghi chú đó do agent chuẩn bị, **chưa phải là bạn đã đọc** — file này là danh sách để bạn tự kiểm.

## 1. Cách đọc mỗi bài (20–30 phút)

Đọc abstract → Figure 1 → hình phương pháp → phần limitation/experiments. Ghi đúng 5 ô (trùng cột matrix):

| Ô | Câu hỏi |
|---|---|
| Representation | Node là gì, edge là gì? Có pose 3D / identity xuyên thời gian không? |
| Construction | Graph do người cho, học từ nhãn, hay tính hình học từ RGB-D? |
| Assumptions | Cần nhãn đóng? camera pose? cảnh tĩnh? |
| Application + evaluation | Dùng để làm gì, đo bằng metric nào (recall@K ≠ task success)? |
| Limitation → liên hệ π0 | Hạn chế này có giải thích được lỗi nào của chúng ta không? |

## 2. Danh sách đọc (6 bài scene graph + 2–3 bài action generation)

| # | Bài | Link | Đọc để lấy gì |
|---|---|---|---|
| 1 | Johnson 2015 — Image Retrieval using Scene Graphs (2D) | [CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Johnson_Image_Retrieval_Using_2015_CVPR_paper.html) | Graph như **ngôn ngữ truy vấn** grounded vào box; không có identity bền vững |
| 2 | Zellers 2018 — Neural Motifs (2D) | [arXiv](https://arxiv.org/abs/1711.06640) | Frequency baseline: predicate đúng có thể do **bias đồng xuất hiện nhãn**, không phải grounding — song song với "position beats identity" của π0 |
| 3 | Armeni 2019 — 3D Scene Graph (3D) | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Armeni_3D_Scene_Graph_A_Structure_for_Unified_Semantics_3D_Space_ICCV_2019_paper.html) | Graph **phân cấp theo metric** (building → room → object → camera) |
| 4 | Gu 2024 — ConceptGraphs (3D) | [project](https://concept-graphs.github.io/) | **Open-vocab**, gộp nhiều khung RGB-D thành object 3D; cách *sinh* graph từ perception cho Part 4 |
| 5 | Jiang 2025 — RoboEXP | [PMLR](https://proceedings.mlr.press/v270/jiang25c.html) | Graph **action-conditioned**, cập nhật khi robot tương tác — mẫu cho phần "update" |
| 6 | Rana 2023 — SayPlan | [PMLR](https://proceedings.mlr.press/v229/rana23a.html) | Graph 3D cho **LLM planning** + feedback; graph ở tầng plan, không ở tầng action |
| (opt) | Hughes 2022 — Hydra | [RSS](https://www.roboticsproceedings.org/rss18/p050.html) | Xây graph phân lớp **online** |
| (opt) | Ji 2020 — Action Genome | [arXiv](https://arxiv.org/abs/1912.06992) | Quan hệ người–vật **thay đổi theo thời gian** (video) |

**Bắt buộc lướt abstract** (công trình đã kết hợp graph + VLA — để **không claim "đầu tiên"**):
[GraSP-VLA](https://arxiv.org/abs/2511.04357) ·
[GF-VLA](https://arxiv.org/abs/2508.05342v2) ·
[SceneDiver](https://arxiv.org/abs/2606.04046).

Action generation:
[π0](https://arxiv.org/abs/2410.24164) (phần action expert + flow matching) ·
[Diffusion Policy](https://arxiv.org/abs/2303.04137) ·
[Octo](https://arxiv.org/abs/2405.12213) (head diffusion nhỏ) ·
nền tảng: [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), [Flow Matching](https://arxiv.org/abs/2210.02747).

Thứ tự gợi ý: π0 → Diffusion Policy → 2 → 4 → 5 → 1 → 3 → 6 → 3 abstract graph+VLA.

## 3. Bốn trục so sánh để viết Part 1 (thay vì tóm tắt từng bài)

- **Representation:** 2D (1, 2) = node grounded vào box, chỉ trong một ảnh; 3D (3, 4) = object có vị trí metric, hợp nhất qua nhiều view.
- **Construction:** học từ nhãn (2) → kế thừa bias; tính hình học / foundation model (4) → lỗi nằm ở bước association. Graph đúng cú pháp ≠ grounded đúng.
- **Time:** ảnh tĩnh (1, 2) vs. video (Ji) vs. map online (Hydra) vs. cập nhật theo hành động (5). Không cái nào tự nó đảm bảo identity qua một lần swap.
- **Use:** retrieval/recall (1, 2) vs. planning (6) vs. exploration (5) vs. điều khiển VLA (GraSP-VLA, GF-VLA). Metric downstream cần là **task success theo đúng identity**.

Câu hỏi mở nên đặt ra: cần node/edge tối thiểu nào để phân biệt hai vật giống hệt; graph thể hiện bất định ra sao khi bị che khuất; khi vật bị di chuyển thì relation cũ bị vô hiệu thế nào; gain của oracle graph còn lại bao nhiêu khi graph do perception tạo ra.

## 4. Diffusion vs flow matching — 4 điểm phải nắm

Ký hiệu: điều kiện $c=(o,\ell)$, action chunk $A\in\mathbb R^{H\times d}$, noise $\epsilon\sim\mathcal N(0,I)$.

| | Diffusion (DDPM/DDIM, Diffusion Policy, Octo) | Flow matching (π0) |
|---|---|---|
| **1. Mục tiêu học** | Dự đoán noise: $\lVert\epsilon-\epsilon_\theta(A_k,k,c)\rVert^2$ | Hồi quy vận tốc: $\lVert v_\theta(A^\tau,\tau,c)-(\epsilon-A)\rVert^2$ |
| **2. Đường nội suy** | Schedule noise: $A_k=\sqrt{\bar\alpha_k}A_0+\sqrt{1-\bar\alpha_k}\,\epsilon$ | Đường thẳng: $A^\tau=\tau\epsilon+(1-\tau)A$ |
| **3. Suy luận** | Khử nhiễu ngược nhiều bước (DDPM, có thêm noise); DDIM ít bước hơn, $\eta=0$ thì tất định | Tích phân ODE; `pi0_libero`: **10 bước Euler** từ $\tau=1$ về 0 |
| **4. Kiến trúc** | Encoder quan sát + denoiser (U-Net/transformer); Octo: backbone transformer + **head DDPM nhỏ** | VLM pretrained + **action expert riêng**, nối với VLM qua attention |

Chi tiết của checkpoint đã chạy (từ code openpi đã pin, xem [PIPELINE.md](docs/reproduction/PIPELINE.md)):
$\tau\sim\mathrm{Beta}(1.5,1)$ co về $[0.001,0.999]$; chunk $50\times7$; thực thi 5 action rồi quan sát lại.
Latency `client.infer` đo được ~93 ms (gồm cả mạng/client, **không** chứng minh FM nhanh hơn diffusion).

Tự kiểm với paper gốc:
- [ ] Quy ước thời gian của π0 trong paper so với code (trong code $\tau=1$ là noise). Hai bên có thể ngược chiều nhau — ghi rõ bạn dùng quy ước nào.
- [ ] Kích thước VLM và action expert, cơ chế attention mask giữa prefix (ảnh + text) và action tokens.
- [ ] Ý chính: cả hai đều là mô hình sinh có điều kiện, đều biểu diễn được nhiều mode. Khác nhau ở **target** và **động lực lấy mẫu**, không phải "ngẫu nhiên vs tất định".
- [ ] (1 câu) Autoregressive token như [FAST](https://arxiv.org/abs/2501.09747) là một nhánh thứ ba, không phải diffusion hay FM.

## 5. Ý tưởng Part 4 — scene graph sửa đúng lỗi đã đo

**Lỗi nhắm tới** (số từ [README](README.md), [results/](results/)): π0 thiếu một liên kết *hiện thời*, *tường minh* giữa thực thể trong câu lệnh, identity/pose của vật, và tiến độ nhiệm vụ.

| Lỗi | Bằng chứng | Biến graph sửa nó |
|---|---|---|
| Position > identity | OBJ-swap 0/9 (ctrl 7/9), S2 0/6 (ctrl 6/6) | con trỏ `target → node` + pose hiện tại |
| Đích open-loop | dời đích: 1/6; 3 lần đặt vật trong ≤1 cm quanh đích cũ | con trỏ `destination → node` + pose hiện tại |
| Không đọc tiến độ | LONG-undo 0/9, LONG-pre 3/9 | predicate sub-goal + lịch sử sự kiện |
| Sửa text không đủ | OBJ-lang 8/9 nhưng OBJ-swap 0/9 | → cần giao diện **không gian** (mask/token), không chỉ viết lại câu |

**Graph chứa gì.** Node: `{id, class, pose 6D, mask ở agent cam + wrist cam}`. Edge hình học: `on / in / next_to / between`, tính từ pose 3D theo ngưỡng. Lớp nhiệm vụ: vai trò `target`, `destination`; danh sách sub-goal dạng `In(soup, basket)` có trạng thái và log sự kiện (thời điểm đạt / bị undo).

**Sinh và cập nhật.** (a) Oracle từ simulator = cận trên, ghi rõ là *privileged information*. (b) Perception: detector/segmenter open-vocab + depth → object 3D (kiểu ConceptGraphs), track identity qua các khung. Cập nhật mỗi lần replan (5 action); mỗi khi predicate đổi trạng thái thì ghi một sự kiện (kiểu RoboEXP).

**VLA dùng graph thế nào** (từ rẻ đến đắt):
- **A. Không cần train — sub-goal monitor:** graph kiểm predicate. Khi một sub-goal chưa đạt hoặc bị undo, prompt lại bằng câu lệnh một-vật cho đúng sub-goal đó. Nhắm LONG.
- **B. Visual prompt:** tô mask target/destination lên ảnh đầu vào. Có thể cần fine-tune.
- **C. Graph tokens:** thêm token cho node target/destination (pose + embedding) vào prefix của action expert, fine-tune LoRA trên demo LIBERO **có augmentation hoán vị vị trí**. Nếu không augment, dữ liệu train luôn gắn chặt vị trí với identity, nên mô hình có thể bỏ qua token.

**Giả thuyết.** H1 (C): OBJ-swap và GOAL-move tăng lên gần mức control, trong khi các control và E0 không giảm. H2 (A): LONG-undo > 0/9.

**Thách thức.** Oracle không chứng minh perception robust. Hai vật giống hệt và che khuất gây lỗi identity. Ngưỡng của quan hệ hình học phải chọn. Latency tăng. Policy có thể phớt lờ input mới.

**Kế hoạch đánh giá.** Dùng lại đúng các manifest Spatial/cross-suite đã có, cùng noise ghép cặp với E0. Các điều kiện:
`baseline | +oracle graph | +perceived graph | text-only rewrite | +graph sai (target bị hoán đổi)`.
Điều kiện *graph sai* chứng minh policy thực sự dùng graph: nếu nó đi theo graph sai, nó đang đọc graph.
Metric: success; vật đầu tiên được nhấc có đúng identity không; khoảng cách tới đích cũ và đích mới; tỉ lệ khôi phục sub-goal.

**Prototype nếu còn thời gian:** chỉ làm A với graph oracle trên 18 episode LONG-undo/LONG-pre. Không phải train, vài giờ trên molab. Rủi ro: π0 có thể không làm theo câu lệnh con trong cảnh Long, nhưng kết quả nào cũng báo cáo được.

## 6. Khung report (≤8 trang) → nguồn trong repo

| Mục | Nội dung | Lấy từ |
|---|---|---|
| 1. Study | 4 trục ở §3 + bảng §4 + matrix (phụ lục) | file này, `../study/` |
| 2. Reproduce | Pipeline, loss, sampling, metric; setup và deviations | [PIPELINE](docs/reproduction/PIPELINE.md), [SETUP](docs/reproduction/SETUP.md), [DEVIATIONS](docs/reproduction/DEVIATIONS.md) |
| 3. Analyse | 4 điểm mạnh / 4 hạn chế, cặp GIF thành công/thất bại | [README](README.md), [results/spatial](results/spatial/README.md), [results/cross_suite](results/cross_suite/README.md) |
| 4. Improve | §5 | file này |
| Caveats | n nhỏ; E0 là pilot 50 episode; T1 không phải phép thử memory sạch | [README § Caveats](README.md#caveats) |

Ghi nhất quán trong report: E0 Spatial có 48/50 (lần chạy cũ) và 49/50 (lần rerun), chọn một và ghi rõ; Goal "1/6" là tập con dời đích, toàn bộ GOAL-move là 4/9.
