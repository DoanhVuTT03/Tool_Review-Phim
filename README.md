# Tool-Edit Review Phim Studio (A–Z)

Công cụ tự động hóa quy trình edit video Review Phim (Thuyết minh & Lồng tiếng) tối ưu cho Dual-GPU Laptop (`Intel Quick Sync` + `NVIDIA CUDA`) và tự động nâng công suất trên PC cấu hình cao (`NVIDIA NVENC` + RTX).

## Cài đặt (lần đầu)
Yêu cầu: Windows 10/11, **Python 3.11**, Microsoft Edge hoặc Google Chrome.

**Cách nhanh:** nhấp đúp **[setup.bat](setup.bat)** — tự tạo `.venv`, cài thư viện, sửa xung đột `onnxruntime`, tải `ffmpeg`/`ffprobe`/`aria2c` vào `bin/` nếu chưa có. Chạy lại bao nhiêu lần cũng được (ví dụ khi AI OCR báo chạy CPU).

Cài tay (tương đương):
1. Tạo môi trường và cài thư viện (chạy trong thư mục tool):
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   rem rapidocr-onnxruntime kéo theo gói "onnxruntime" thường và gói này ĐÈ LÊN bản DirectML -> sửa lại:
   .venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-directml opencv-python
   .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps onnxruntime-directml opencv-python-headless
   ```
   > Kiểm tra: `.venv\Scripts\python.exe -c "import onnxruntime as o; print(o.get_available_providers())"` phải có `DmlExecutionProvider`. Nếu không, AI OCR xóa sub chỉ chạy CPU (badge phần cứng trên UI cũng sẽ cảnh báo).
2. Tải **FFmpeg** (bản *essentials* của [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)) và chép `ffmpeg.exe`, `ffprobe.exe` vào thư mục `bin/`.
   Tùy chọn: chép thêm `aria2c.exe` ([aria2](https://github.com/aria2/aria2/releases)) vào `bin/` để tải video đa luồng nhanh hơn.
3. (Tùy chọn) Dịch SRT bằng tài khoản Antigravity: mở app **Antigravity** (đã đăng nhập Google) khi dịch, hoặc nhập Gemini API Key ở Tab Dịch SRT.

## Tối ưu tự động theo máy
Tool tự nhận phần cứng (`/api/hardware`) và chọn cấu hình — không cần chỉnh tay:

| | Laptop Dual-GPU (i7 + Iris + MX230 2GB) | PC NVIDIA (vd. i5-12400F + RTX 3060 12GB) |
|---|---|---|
| Giải mã + ghép lớp (sub, logo, nền xanh, dải xóa sub) | CPU | **GPU**: NVDEC → `overlay_cuda` → NVENC, khung hình không rời VRAM |
| Nén video | Intel QSV `h264_qsv` | NVENC `h264_nvenc` preset **p2** (nhanh ~1.6× p4, cùng chất lượng) |
| AI OCR dò sub | CPU + Iris + MX230 (DirectML) | **PyTorch CUDA**, dò theo lô (cùng model PP-OCR, kết quả giống ONNX) |
| Tái tạo nền (xóa chữ) | 4 luồng CPU (Navier-Stokes) | **PyTorch CUDA** theo lô 48 frame (lấy pixel thật từ frame sạch lân cận, nếu không được thì lấp mượt) |
| A–Z hàng loạt | 1 render AI | 1 render AI (GPU đã bận 80–90%), 2 render thường |

- PyTorch CUDA được `setup.bat` cài tự động khi máy có card NVIDIA (~2.5 GB). Không có PyTorch thì tool vẫn chạy: OCR qua DirectML, tái tạo nền bằng CPU.
- **Tải video (⬇ Link YT/Bili):** chỉ tải video + phụ đề gốc về danh sách (chưa dịch / xóa sub / xuất). YouTube tải trực tiếp bằng yt-dlp (chạm trần băng thông ~100 Mbps: 12 MB/s; aria2c bị YouTube bóp còn 3.8 MB/s nên chỉ dùng cho BiliBili). Ưu tiên VP9 rồi H.264 ≤1080p tải thẳng (video 8 tiếng: 3.7 GB thay vì 8.1 GB "Premium" HLS); không chọn AV1 vì giải mã CPU chậm (285 fps) làm nghẽn bước xóa sub. Đóng cửa sổ thì vẫn tải ở nền, mở lại tự nối tiếp; có nút ⏹ Dừng tải. Dọn rác SRT bằng AI chạy 6 lô song song.
- **Xóa chữ logo / watermark (lớp "Xóa chữ logo"):** thêm bao nhiêu khung tùy ý (vd. "本作品含AI生成内容" ở góc), AI dò chữ trong từng khung và xóa (hoặc chọn làm mờ). Mọi vùng (sub + chữ logo) dùng chung 1 lần giải mã video. Tái tạo nền GPU nội suy theo chiều dọc nên giữ thẳng mép tường / ranh giới màu.
- **Chữ tên kênh (lớp "Tên kênh"):** font, cỡ, màu chữ / viền / nền, bo góc + hiệu ứng lặp (nhịp thở, bồng bềnh, nhấp nháy, ánh sáng lướt qua, trượt vào, gõ chữ).
- **Hiệu ứng phụ đề:** mờ dần, trượt lên, bật nảy, phóng to, gõ chữ, nhòe → nét, rơi nảy, trượt ngang (+ mờ dần khi hết câu). Dựng từ đúng ảnh phụ đề nên giữ nguyên kiểu chữ / khung; xem trước ngay trên video.
- **Style Guide** đọc **cả file SRT** (trước chỉ lấy mẫu ~250 dòng).
- **A–Z hàng loạt:** trong lúc chờ Dịch SRT + Tạo Voice (chờ mạng, máy rảnh), tool **quét OCR sub gốc trước** → bước render chỉ đọc cache.
- Đo trên RTX 3060 + i5-12400F, video 1080p 5 phút có hardsub, xóa sub AI + xuất video (chưa có cache OCR): bản gốc **43.8 s (206 fps, 6.9× thời gian thực)** → **19.0 s (475 fps, 15.8×)**, tức **nhanh ~2.3×**; phim 1 tiếng ≈ 4 phút. CPU chỉ còn giải mã dải ROI cho AI.
- Đồ thị GPU tự lùi về đồ thị CPU nếu video không giải mã được bằng NVDEC hoặc kích thước khung ra sai; chế độ *Làm mờ bằng khung* luôn dùng đồ thị CPU.
- **LaMa:** model LaMa ONNX công khai không chạy được trên DirectML (lỗi MatMul) và trên CUDA mất ~150 ms/frame (chậm hơn chế độ nhanh hàng chục lần). Chọn LaMa mà không có GPU chạy được thì tool tự dùng chế độ nhanh.

> Thư mục `data/` (video, project, cache OCR, log) và `bin/` không nằm trong repo — tool tự tạo `data/` khi chạy.

## Khởi động
- Nhấp đúp **[run.bat](run.bat)** (hoặc shortcut *Tool-Edit Review Phim*): tool chạy **ẩn, không hiện màn hình cmd**, chỉ mở cửa sổ giao diện tại `http://127.0.0.1:8765`.
  - Bấm lại khi tool đang chạy → chỉ mở thêm cửa sổ giao diện, không chạy server thứ 2.
  - Đóng cửa sổ giao diện → server tự tắt sau ~5 phút (không tắt khi đang có job dịch / tạo voice / xuất video).
  - Log server: `data/logs/server.log`.
  - Muốn shortcut không lóe cmd: tạo shortcut tới `.venv\Scripts\pythonw.exe` với tham số `"<thư mục tool>\launcher.pyw"`.
- **[run_debug.bat](run_debug.bat)**: chạy kiểu cũ có màn hình cmd + tự reload khi sửa code (dùng khi phát triển / xem lỗi).

## Luồng làm việc 4 Bước

### 1. Thiết lập Cấu hình & Preview trực quan (Tab 1)
- Thêm 1 hoặc nhiều video vào cột trái (tự nhận file `.srt` cùng tên nếu đặt cạnh nhau).
- Click vào video bất kỳ để mở màn hình Preview:
  - **Phụ đề (Subtitle)**: Chọn từ 300+ font hệ thống hoặc nạp `.ttf/.otf`, chỉnh cỡ chữ, đậm, màu chữ, viền, màu nền, độ mờ nền, **bo góc (`border-radius`)**, kéo vị trí & **resize chiều rộng/chiều cao khung độc lập**.
  - **Logo Kênh**: Chọn 1 file logo hoặc chọn **thư mục chứa nhiều logo** để click chọn nhanh theo kênh.
  - **Video Nền Xanh (Like/Subscribe)**: Xóa phông xanh trực tiếp ngay trên Preview, kéo W×H độc lập, chọn thời điểm xuất hiện hoặc lặp lại.
  - **Xóa sub gốc / Làm mờ**:
    - *Làm mờ bằng khung*: Kéo W×H tự do, siêu nhanh.
    - *Xóa sub sạch (AI Clean Sub)*: Khoanh vùng ROI + OCR (PP-OCR detector chạy song song trên CPU + GPU qua DirectML) tự dò đúng frame có chữ, tìm chính xác frame bắt đầu/kết thúc của từng câu sub + Temporal Background Propagation (tùy chọn nạp thêm `data/models/lama_fp32.onnx` cho LaMa AI). Xóa sub chạy **cùng một lượt** với bước xuất video (1 lần giải mã + 1 lần nén), kết quả OCR được cache để xuất lại không phải quét lại. Các frame không có sub giữ nguyên 100% pixel gốc.
  - **Preset Kênh**: Bấm **Lưu Preset** trên thanh trên cùng để tái sử dụng cho hàng loạt video.

### 2. Dịch SRT qua Antigravity (Tab 2)
- Chọn ngôn ngữ gốc (mặc định Trung Quốc) và ngôn ngữ đích (danh sách ngôn ngữ toàn cầu).
- Chọn Model + Cấp độ tư duy (`Low`, `Medium`, `High`, `Max`).
- Tự động đọc luật dịch từ `gem.txt`:
  1. **Pha 1 — Phân tích Style Guide**: Chốt Topic, Context, Nhân vật/Xưng hô, Văn phong và lưu thành `<ten_video>_style_guide.md` (có thể sửa trực tiếp trên UI).
  2. **Pha 2 — Mở phiên chat mới & Dịch theo lô**: Kèm ngữ cảnh nối mạch giữa các lô, kiểm soát độ dài câu `±15%` chuẩn timing Voice AI, **lưu checkpoint sau mỗi lô** để chạy tiếp nếu bị gián đoạn. Dịch **nhiều luồng song song** (số luồng tùy chỉnh 1–16).
- Nút **⏹ Dừng Tiến Trình** dừng ngay mọi bước (dịch / voice / xóa sub / xuất video); chạy lại sẽ tiếp tục từ checkpoint.
- **⏱ Ép câu dịch vừa khung thời gian** (mặc định bật, mọi ngôn ngữ): mỗi dòng gửi AI kèm *số âm tiết tối đa* (tiếng Trung/Nhật/Hàn/Thái: số ký tự),
  tính từ thời lượng câu trên video + tốc độ đọc tự nhiên của ngôn ngữ đích (Filipino đo trên Edge: 4,8 âm tiết/giây), chừa 0,3 s nghỉ giữa 2 câu.
  Dịch xong, câu nào vẫn dài được AI rút gọn (giữ ý, tên riêng, danh xưng). Nút **✂ Rút gọn câu tràn khung** làm việc này cho bản dịch có sẵn.
  *Lý do*: video 512 (Trung → Filipino) có câu dịch dài gấp ~2 lần chỗ trống → voice bị tua nhanh và đọc dồn sang câu sau.

### 3. Tạo Voice (Tab 3)
- Liệt kê các giọng miễn phí của **CapCut / TikTok** đa ngôn ngữ (Tiếng Việt `BV074_streaming`, `BV075_streaming`, Tiếng Anh, Trung, Nhật, Hàn...) + **300+ giọng miễn phí Edge-TTS Neural** dự phòng tự động.
- Hỗ trợ **Nghe thử giọng** trực tiếp và tạo voice song song có **cache trên ổ đĩa** (chỉnh sửa lại video không cần gọi lại TTS).
- **✂ Tự rút gọn câu đọc dài quá khung** (mặc định bật): đo độ dài thật từng câu voice; câu nào cần tua quá mức được AI rút gọn rồi đọc lại (chỉ các câu đó).
- **Higgs TTS 3 — clone giọng mẫu** (RTX 3060 12GB; máy chủ sglang-omni trong WSL Ubuntu, dùng chung cài đặt `D:\Download\Model\higgs3_*`):
  1. Giọng mẫu: chọn file trong thư mục (vd `D:\Download\Youtube\Voice\Mau`, file `.txt` cùng tên = lời). **🧼 Lọc sạch** bằng Mel-Roformer
     (tách nhạc nền) + UVR-DeEcho-DeReverb (khử vang/ồm) + Mel-Roformer denoise (khử tạp âm), đo DNSMOS trước/sau từng bước, cắt đoạn
     8–15 s gồm câu trọn vẹn. *Tự động*: chỉ giữ bước làm giọng sạch hơn rõ; *Lọc mạnh*: luôn chạy đủ 3 bước. Nút **🎧 Lọc & nghe thử** để nghe trước/sau.
  2. Tạo voice **48 câu cùng lúc** (mỗi câu tự mang câu chữ + giọng mẫu + seed riêng; ~27 giây âm thanh / giây), câu dài trước.
  3. Tắt máy chủ → **Whisper large-v3-turbo** nghe lại mọi câu (gộp ~10 câu / cửa sổ 28 s, ~25 câu/giây), so từng từ: thiếu từ, mất từ cuối,
     đọc sai, dính/tách từ, thừa/lặp từ, dài/ngắn bất thường, chạm trần độ dài, dài quá khung. Bỏ qua khác biệt chỉ do chính tả
     (`nang`/`ng`, `ganun`/`ganoon`, tên phiên âm `qi`/`ki`...). Câu đọc sai được **tự tạo lại** (0–3 lần, giữ bản tốt nhất).
  4. **📋 Bảng kiểm tra voice**: câu gốc (sửa được) · Whisper nghe được (tô chữ lệch) · lý do · ▶ nghe · chọn bản · ↻ tạo lại từng câu /
     câu đã chọn / tất cả · ✓ đánh dấu OK · **✔ Xong kiểm tra → chạy tiếp**. Chạy A–Z sẽ *tạm dừng* sau bước voice nếu còn câu nghi ngờ.
  - Higgs chiếm ~12 GB VRAM: khi chọn Higgs, **quét OCR / xóa sub / xuất video chỉ chạy sau khi tạo voice xong**; tool tự tắt máy chủ Higgs
    và giải phóng VRAM trước các việc GPU khác (khóa GPU `app/services/gpu_gate.py`).
  - Môi trường âm thanh riêng: `D:\Download\Model\envs\giong_sach` (audio-separator + faster-whisper), hoặc biến môi trường `TOOL_AUDIO_PYTHON`.

### 4. Edit Thuyết minh / Lồng tiếng & Chạy A–Z Hàng Loạt
- **Thuyết minh (`narration`)**: Tự cắt khoảng lặng đầu/cuối mỗi câu, mượn khoảng nghỉ phía sau trước khi tua nhẹ giữ cao độ (`librubberband`), tự động **Auto-Ducking** hạ nhạc/tiếng gốc mượt mà khi giọng đọc cất lên.
- **Lồng tiếng (`dubbing`)**: Cắt khoảng lặng + ép chặt đúng khung thời gian `[start, end]` từng thoại để không đè cảnh/nhân vật, mặc định tắt âm thanh gốc.
- **Chạy A–Z Hàng Loạt (`🚀 Chạy A–Z Hàng Loạt`)**: Chạy gối đầu song song I/O (Dịch + Voice) + AI GPU + Video Encoder (`h264_qsv` / `h264_nvenc`).
