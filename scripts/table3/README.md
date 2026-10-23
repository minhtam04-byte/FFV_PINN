# Bảng 3: hốc xốp đồng nhất (Nu trên vách nóng, 18 case)

Tham chiếu: Liu, Wang, Jv & Guo (2021), *Commun. Comput. Phys.* 29:265, Table 3 (`references/dugks_porous.pdf`).
Bảng đó có ba cột tham chiếu: Ref. [22], Ref. [39] và DUGKS. Các giá trị nằm trong `_common.py`.

## Thiết lập

- Hốc vuông xốp đồng nhất: `cases/porous_cavity/config/base.py`, ghi đè tham số qua dòng lệnh, không có config riêng.
- Pr = 1, ν_e = ν, F_ε = 1.75/√(150 ε³) (Eq. 2.4 của bài): 0.564809 với ε = 0.4 và 0.307444 với ε = 0.6.
- Da ∈ {1e-2, 1e-4, 1e-6}, mỗi Da có 3 giá trị Ra (xem `REF`), ε ∈ {0.4, 0.6}.
- Scheme `ffv_compact`, h = 1/100, batch 4096, có PTS và GradNorm như trong `base.py`.
- Nu báo cáo là trung bình ± độ lệch chuẩn của `metric/Nu_hot` trong **10k bước cuối**, không phải giá trị ở một bước.
  Một snapshot riêng lẻ có thể nhiễu tới khoảng 3%.

## Các giai đoạn

Run nằm trong `outputs/porous_cavity/porous_cavity_<tiền tố>_<da>_<e>_Ra<…>/`, log nằm trong `outputs/porous_cavity/logs/`.

| Script | Tiền tố | Nội dung |
|---|---|---|
| `run_C.sh` | `tab3C_*` | 18 case, 50k bước |
| `run_D.sh` | `tab3D1_*`, `tab3D2_*` | Da 1e-6, ε 0.4: resume tới 100k (D1) và chạy 150k với decay_steps 6000 (D2). D1 tốt hơn ⇒ cần lr cuối nhỏ |
| `run_E.sh` | `tab3E_*` | **Bản dùng cho bài**: chép mỗi run C (hoặc D1) rồi resume tới 100k, giữ lịch lr |
| `run_F_ad.sh` | `tab3F_ad_*` | Baseline AD, 100k bước liền, chỉ khác `diff.scheme=ad` |
| `run_speed.sh` | `speed_*` | Đo tốc độ theo Bảng 2 của Wang et al., 5000 bước mỗi run |

Mọi script `.sh` tự chuyển về gốc repo và chạy lại được an toàn: run đã xong được bỏ qua, run dở được resume.
Thêm `--dry_run` để chỉ in lệnh (với `run_C.sh`, `run_F_ad.sh`, `run_speed.sh`).
Nên chạy trong tmux, ví dụ `tmux new -s tab3 'bash scripts/table3/run_E.sh'`.

## Phân tích

Các script Python chạy được từ bất kỳ thư mục nào, kể cả nút ▶ của VS Code, vì `_common.py` chuyển về gốc repo.
Khi có tham số, chạy trong terminal:

```bash
venv/bin/python scripts/table3/summarize.py tab3E          # -> outputs/porous_cavity/table3_E.csv
venv/bin/python scripts/table3/summarize.py tab3F_ad       # -> table3_F_ad.csv
venv/bin/python scripts/table3/eval_runs.py tab3E_da4_e04 --ref=7.808
venv/bin/python scripts/table3/loss_analysis.py tab3C
venv/bin/python scripts/table3/compare_D.py
venv/bin/python scripts/table3/speed_summary.py
```

- `summarize.py`: tạo bảng chính. Cột `nu_mean_last10k`/`nu_std_last10k` là Nu trong 10k bước cuối. Các cột `*_401` đánh giá
  checkpoint cuối trên lưới 401²: Nu ở vách nóng, vách lạnh, mặt cắt giữa, và sai số khối lượng.
  Các CSV tạo trước 2026-10-23 dùng tên cũ `nu_mean_40_50k`, `nu_final_401` cho cùng nội dung.
- `eval_runs.py`: đánh giá nhanh mọi run có cùng tiền tố, không ghi file.
- `loss_analysis.py`: chẩn đoán hội tụ (độ dốc loss, nhiễu Nu, thành phần loss, trọng số PTS).
- `compare_D.py`: so C, D1, D2.
- `speed_summary.py`: giây/100 bước, bỏ khoảng đo ở bước 1000–1100 (JIT lần đầu của cập nhật GradNorm/PTS).
- `eval_old_present.py`: truy nguồn cột "Present" cũ, cần ổ HDD của PC. Cột đó dùng F_ε = 0.1 nên **không dùng cho bài**.

Đánh giá dùng GPU. Không chạy trong lúc đang đo tốc độ.

## Kết quả (2026-10-23)

- **E** (`ffv_compact`, 100k): cả 18 case lệch ≤ 2.8% so với Ref. [22], ≤ 1.3% so với tham chiếu gần nhất trong ba cột.
  Sai số khối lượng ≤ 0.5% với Da 1e-2 và 1e-4, 1.1–3.4% với Da 1e-6.
- **F** (AD, 100k): tương đương E với Da 1e-2 và 1e-4. Với Da 1e-6, Ra 1e9, AD lệch −9.0% (ε 0.4) và −12.4% (ε 0.6),
  sai số khối lượng 5–8%.
- **Tốc độ** (RTX 3060, giây/100 bước, có PTS): AD 1.443, `ffv_continuous` 1.852, `ffv_compact` 1.712,
  `ffv_compact` với `diff.fuse_pts_stencil=True` 1.206.
