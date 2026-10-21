# Roadmap

Các việc nghiên cứu còn mở cho bài báo. Đánh dấu `[x]` khi xong, ghi ngày và commit liên quan.

## Định hướng đã chốt

- Không đề cập việc reproduce FFV-PINN. Trình bày là *xây dựng trên* sơ đồ FV rút gọn của Wei et al. (2025):
  giá trị mặt lấy trực tiếp từ mạng. Đây là ghi nhận nguồn gốc, không chỉ là "lấy cảm hứng".
- Chỉ ra rằng residual correction loss của FFV-PINN (Eq. 18–20) tương đương PTS với 1/Δτ = a_P cố định;
  BB-PTS của Wang et al. (2026) thay bằng ước lượng tự động. Không cần reproduce code của họ.
- Mọi so sánh làm trong cùng một framework (cùng kiến trúc, optimizer, GradNorm, sampling); mỗi lần chỉ đổi một thành phần.

## A. Lỗi và rủi ro tái lập đã biết (cần sửa code)

- [ ] `notebooks/fluid_porous_layer.ipynb` cell 3: `titles` lệch với `model_benchmark`; `history_benchmark` đảo vị trí 4–5;
      model case 1 (`..._case_1_pts_fix_grid_42_grid60_60`) bị vẽ chung với case 2; mọi ψ_max đều in nhãn "Case 1 Reference".
- [x] 2026-10-20: **Matmul precision** từng là side effect khi import trainer (`fluid_porous_layer` đặt "highest",
      `porous_cavity` train bằng TF32; cùng một checkpoint cho ψ_max 24.49 so với 24.78). Giờ ghi trong
      `config.jax_matmul_precision` và áp dụng bởi `cases.experiment.apply_runtime` (cả khi train lẫn khi nạp lại).
- [x] 2026-10-20: Hai trainer riêng được thay bằng `jaxpi.training.train` qua `cases.experiment.train_case`
      (resume, evaluator, dừng khi gặp NaN). Đã kiểm chứng: tham số sau các bước train giống từng bit với trainer cũ.
- [ ] Notebooks vẫn tự tính ψ_max (`compute_stream_function_and_psimax`, định nghĩa 2 lần, lưới 80 và 120). Thay bằng
      `porous_pinn.evaluators.convection_fields(model, params, n)` (ψ_max, mass imbalance, Nu nóng/lạnh) trên lưới ≥ 401².
- [ ] **Hai config vật liệu khác nhau ở 7 siêu tham số** (Fourier scale, n_x, scheme mặc định, trọng số `T_neumann_bc`,
      `shrink.end_log_drop`), xem `cases/fluid_porous_layer/config/glass.py`. Quyết định khác biệt nào là cố ý trước khi
      so sánh glass với aluminum trong bài báo.
- [ ] README: phương trình động lượng ghi `sqrt(Pr/Ra)(1/ε)∇²u` nhưng code dùng `sqrt(Pr/Ra)∇²u` (khớp Gao et al. với J = 1);
      README ghi "warmup cosine decay" nhưng các config dùng `exponential_decay`. Xác định bên nào đúng.
- [ ] `notebooks/legacy/porous_cavity_train_old.ipynb` dùng scheme `nd`/`can` đã bị xóa khỏi `numerical_pde.py`, nên không chạy được.
      Các run cũ `*_can_*`, `*_nd_*` trong `outputs/` cũng không nạp lại được bằng code hiện tại.

## B. Phương pháp số

- [x] 2026-10-20: `ffv` bỏ nội suy áp suất kiểu CAN (cần ∂p/∂x, ∂p/∂y từ AD) và dùng `(p_e − p_w)/dx`, `(p_n − p_s)/dy`
      như `ffv_continuous`; scheme giờ không còn dùng AD (33.6 ms/bước so với 52.7 trước đó). **Mọi run `ffv` trước ngày này
      (kể cả `verify_cavity_ffv`, `verify_case2_ffv`) dùng áp suất CAN**, nên phải chạy lại trước khi đưa vào bài báo.
- [ ] `ffv` và `ffv_continuous` vẫn khác nhau ở toán tử residual (trừ φ_P·div trong đối lưu; harmonic mean giả định mặt phân cách
      ở giữa P–E so với có trọng số khoảng cách; drag/buoyancy giá trị điểm so với tích phân dưới ô; k theo phương y). Để tách
      ảnh hưởng của cách lấy mẫu, so `ffv_continuous` với `use_continuous_mesh=False` và `True`.

- [x] 2026-10-20: **Phủ vùng sát vách** bằng scheme mới `ffv_compact`: khuếch tán lấy từ giá trị tại mặt (cách P h/2),
      stencil vươn h/2, tâm CV trong [h/2, L−h/2]. Run kiểm chứng (aluminum, 100k bước, 1 seed, đánh giá trên lưới 401²):

      | scheme           | ψ_max            | lỗi khối lượng | Nu nóng / giữa / lạnh  | thời gian |
      |------------------|------------------|----------------|------------------------|-----------|
      | ffv_continuous   | 24.66 (+6.3% LBM) | 4.02%          | 9.04 / 8.54 / 8.33     | 42 phút   |
      | ffv_compact      | 24.76 (+6.7% LBM) | 0.83%          | 8.80 / 8.92 / 8.85     | 33 phút   |

      Bảo toàn khối lượng và năng lượng tốt hơn ~5–15 lần, nhưng **ψ_max gần như không đổi**: sai lệch ψ_max còn lại không do
      vùng sát vách. Nghi phạm tiếp theo: sai số rời rạc theo h (mục C) và ε chưa rõ của case 4.3.
- [ ] Luận điểm cho ffv_continuous trong bài: (1) kích thước CV h tách khỏi vị trí mặt phân cách nhờ phần thể tích dưới ô
      chính xác; (2) các CV ngẫu nhiên chồng lấn ≈ dạng yếu với hàm thử hộp độ rộng h, tích phân Monte Carlo (liên hệ
      hp-VPINN, control-volume PINN). Nêu thẳng điểm yếu: mất tính bảo toàn toàn cục rời rạc, rồi đo |ψ(y=1)| và
      Nu_hot so với Nu_cold. Lý do "h là số lẻ" không phải lập luận khoa học, không dùng.
- [ ] Lưới cố định: chọn n_x chia hết cho 3 (60, 81, 120) để mặt phân cách 1/3, 2/3 trùng mặt ô
      (`compute_ffv_pde_residuals` giả định mặt phân cách nằm giữa P và E).
- [ ] `sol_net` dùng trọng số [0.5, 0.125×4] → φ_P + h²/32·∇²φ; trung bình ô đúng là [1/3, 1/6×4] (φ_P + h²/24·∇²φ).
- [ ] Ablation PTS: tắt / τ = a_P cố định (≈ RC loss của FFV-PINN) / BB thích ứng.
- [ ] Docstring của `can_ffv` nói nội suy Hermite O(h⁴), nhưng `tests/test_fv_schemes.py` đo được bậc 2.00 cho toàn sơ đồ
      (div từ giá trị tại mặt chỉ đạt bậc 2). Sửa docstring, hoặc bỏ `can_ffv` nếu không đem lại lợi ích.

## C. Kiểm chứng cho bài báo

- [x] 2026-10-21: **Đọc Beckermann et al. (1988), `references/JFM1988.pdf`.** Case 4.3 = thí nghiệm 12 (glass) và 14 (Al), cấu hình 2.
      - Cột cuối Bảng 1 là hệ số quán tính Ergun C = 0.554 (Gao gọi là Fe), **không phải độ rỗng**. Suy ε từ Eq. (9)–(10):
        C = 1.75 / (√175 · ε^1.5) ⇒ **ε = 0.385** (kiểm chứng: Eq. (9) cho lại Da = 1.368e-5 / 1.533e-5, khớp Bảng 1 trong 0.2%).
        Config đang dùng ε = 0.4.
      - **ψ_max 16.4 / 23.9 là kết quả số của Beckermann** (SIMPLER, lưới **26×26** dãn về phía vách; 50×50 "không cải thiện
        đáng kể"), không phải số đo. Gao viết "same grid 120 × 120 as Ref. [11]" là sai.
      - **Mô hình khác Gao/của ta.** Eq. (14) của Beckermann: trong vùng xốp *không có* số hạng quán tính (u·∇)u, Brinkman với
        μ_eff = μ, không có ε. Gao (và code này), sau khi chia cho ε: quán tính (1/ε²)(u·∇)u, Brinkman (1/ε)ν∇²u. Darcy, Forchheimer,
        buoyancy, năng lượng thì như nhau. Lớp Brinkman tại mặt phân cách: √Da = 0.0039 (Beckermann) so với √(Da/ε) = 0.0062 (ta).
      - Số liệu thực nghiệm định lượng chỉ có cho glycerin (thí nghiệm 1–6, ô lớn, Ra = 9.52e5, Pr = 8356.2): profile nhiệt độ tại
        3 độ cao (Hình 2–4). Với cấu hình 2: thí nghiệm 3 (glass, R_k = 2.259) và 4 (Al, R_k = 17.10), Da = 2.248e-6 / 2.518e-6,
        C = 0.565 (ε = 0.380). Thí nghiệm 12/14 chỉ có ảnh giao thoa (định tính).
- [x] 2026-10-21: **Độ nhạy theo ε và theo dạng phương trình tại mặt phân cách** (aluminum, `ffv_compact`, h = 1/80, 3 seed,
      lưới đánh giá 401²; hình: `outputs/fluid_porous_layer/figures/psi_max_interface_form_sensitivity.pdf`):

      | trường hợp          | ψ_max          | so với LBM / Beck. | Nu_mid        | lỗi khối lượng % |
      |---------------------|----------------|--------------------|---------------|------------------|
      | gao, ε = 0.400      | 24.730 ± 0.040 | +6.5% / +3.5%      | 8.881 ± 0.062 | 0.68 ± 0.25      |
      | gao, ε = 0.385      | 24.888 ± 0.046 | +7.2% / +4.1%      | 8.939 ± 0.065 | 0.77 ± 0.10      |
      | hirata, ε = 0.385   | 21.043 ± 0.039 | −9.3% / −12.0%     | 7.310 ± 0.019 | 0.35 ± 0.07      |

      ε 0.4 → 0.385: +0.6% (nhỏ, nhưng > 3σ). Dạng tại mặt phân cách (gao → hirata): **−15.5% ψ_max, −18% Nu**. Hai tham chiếu
      (23.21, 23.9) nằm giữa hai dạng: sai lệch giữa các nghiên cứu nằm trong độ nhạy theo cách mô hình hóa mặt phân cách.
- [ ] Quyết định dạng phương trình dùng làm kết quả chính của bài báo (gao để so với Gao 2017, hirata nhất quán với lý thuyết
      trung bình thể tích) và cách trình bày độ nhạy này.
- [ ] (Hạ ưu tiên) Biến thể "beckermann" (bỏ quán tính trong vùng xốp, μ_eff = μ): đo được quán tính và Brinkman dư chỉ
      đáng kể trong lớp Brinkman tại mặt phân cách; biến thể hirata đã trả lời câu hỏi về độ nhạy theo điều kiện tại mặt phân cách.
- [x] 2026-10-21: Số hóa Hình 3 của Beckermann (thí nghiệm 3 và 4, glycerin): `reference_data/beckermann1988_fig3.csv`;
      config `exp3_glycerin_glass.py`, `exp4_glycerin_aluminum.py`; script `scripts/plot_beckermann_fig3.py`.
- [ ] **Case glycerin chưa hội tụ** (thí nghiệm 3, 100k bước: loss ~6–8e-3, lỗi khối lượng 12–19%, profile T gợn sóng), nên so sánh
      với Hình 3 chưa có kết luận (RMS Δθ ≈ 0.06–0.07). Pr = 8356: vận tốc thực ~ 1/√Pr theo thang rơi tự do và hệ số Darcy
      ~1.6e4. Thử: thang vận tốc theo độ nhớt cho Pr ≫ 1, train lâu hơn / lịch PTS khác. Run thí nghiệm 4 đã bị bỏ qua.
- [ ] **Ưu tiên cao nhất (sau kết quả hội tụ theo h):** lời giải tham chiếu hội tụ lưới (≥ 480²) cho *cùng hệ phương trình và
      tham số* bằng jax-dugks (`venv/src/jax-dugks`) hoặc FVM SIMPLE, để biết sai lệch ψ_max ≈ 24.70 so với 23.21 đến từ PINN
      hay từ khác biệt mô hình (ε, cách viết phương trình) so với Gao et al.
- [ ] FVM cổ điển với cùng sơ đồ, cùng h: FFV-PINN(h) − FVM(h) = sai số tối ưu; FVM(h) − tham chiếu = sai số rời rạc.
- [x] 2026-10-21: Hội tụ theo h (aluminum, `ffv_compact`, CV ngẫu nhiên, 100k bước, 3 seed; `scripts/sweep.py --prefix=hconv`;
      đánh giá trên lưới 401², precision "highest"):

      | h     | ψ_max (mean ± std) | so với LBM / Beck. | lỗi khối lượng % | Nu_mid        | \|Nu nóng − lạnh\| % |
      |-------|--------------------|--------------------|------------------|---------------|----------------------|
      | 1/40  | 25.018 ± 0.073     | +7.8% / +4.7%      | 1.53 ± 0.07      | 9.042 ± 0.083 | 1.80 ± 1.94          |
      | 1/60  | 24.772 ± 0.096     | +6.7% / +3.6%      | 0.99 ± 0.24      | 8.966 ± 0.037 | 1.87 ± 1.06          |
      | 1/80  | 24.730 ± 0.040     | +6.5% / +3.5%      | 0.68 ± 0.25      | 8.881 ± 0.062 | 0.56 ± 0.45          |
      | 1/120 | 24.666 ± 0.063     | +6.3% / +3.2%      | 0.43 ± 0.12      | 8.870 ± 0.039 | 0.51 ± 0.10          |
      | 1/160 | 24.707 ± 0.037     | +6.4% / +3.4%      | 0.38 ± 0.16      | 8.833 ± 0.016 | 0.18 ± 0.20          |

      **ψ_max hội tụ từ h ≤ 1/80 về ≈ 24.70 ± 0.05**, tức sai lệch +6.4% so với LBM là hệ thống (≫ dao động giữa các seed,
      ~0.3%) và **không do rời rạc hóa**. Nu_mid vẫn giảm nhẹ theo h (≈ 8.83 ở 1/160); lỗi bảo toàn giảm đơn điệu theo h.
      Loss cuối tăng khi h nhỏ (1.3e-5 → 3.8e-5): tối ưu khó hơn với stencil hẹp.
- [ ] Mọi số liệu chính: mean ± std trên ≥ 5 seed (`scripts/train.py --config.seed=...`).
- [ ] Lưới cố định đã căn mặt phân cách vs ffv_continuous ở cùng h, cùng PTS.
- [ ] Đo trên toàn trường: L2 của u, v, T; Nu trung bình; profile v(x), T(x) tại y = 0.5; so với LBM và Beckermann.
- [ ] Bằng chứng cho luận điểm trung tâm: với AD (aluminum), Nu ở hai vách cân bằng (8.11 / 8.08) nhưng thông lượng nhiệt
      tổng qua mặt cắt giữa (trong lớp xốp) là 38.1, gấp 4.7 lần. AD không giảm gradient T trong vùng xốp theo 1/λ, vì thiếu
      điều kiện liên tục thông lượng. Báo cáo `metric/Nu_mid` cho mọi scheme.
- [ ] Baseline AD-PINN có xử lý mặt phân cách (cPINN/XPINN có ràng buộc liên tục thông lượng, hoặc mạng discontinuity-capturing).
      AD-PINN hiện tại (`porous_pinn/models.py`, nhánh `ad`) không có điều kiện liên tục thông lượng tại mặt phân cách.

## D. Mở rộng

- [ ] Bài toán ngược: xác định Da, k_e/k_f hoặc ε từ vài điểm đo nhiệt độ.
- [ ] Case 4.4 (solid–porous có tường dẫn nhiệt, Bảng 2 của Gao et al.) và Bảng 1 (cavity đồng nhất, Nu).
- [ ] Bảng chi phí: thời gian, GPU, so với LBM/FVM.
