# Tool-Edit Review Phim Studio (A–Z)

Công cụ tự động hóa quy trình edit video Review Phim (Thuyết minh & Lồng tiếng) tối ưu cho Dual-GPU Laptop (`Intel Quick Sync` + `NVIDIA CUDA`) và tự động nâng công suất trên PC cấu hình cao (`NVIDIA NVENC` + RTX).

## Cài đặt (lần đầu)
Yêu cầu: Windows 10/11, **Python 3.11**, Microsoft Edge hoặc Google Chrome.

1. Tạo môi trường và cài thư viện (chạy trong thư mục tool):
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   > `onnxruntime-directml` đã bao gồm CPU + GPU (DirectML). **Không** cài thêm gói `onnxruntime` thường (sẽ ghi đè bản DirectML).
2. Tải **FFmpeg** (bản *essentials* của [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)) và chép `ffmpeg.exe`, `ffprobe.exe` vào thư mục `bin/`.
   Tùy chọn: chép thêm `aria2c.exe` ([aria2](https://github.com/aria2/aria2/releases)) vào `bin/` để tải video đa luồng nhanh hơn.
3. (Tùy chọn) Dịch SRT bằng tài khoản Antigravity: mở app **Antigravity** (đã đăng nhập Google) khi dịch, hoặc nhập Gemini API Key ở Tab Dịch SRT.

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

### 3. Tạo Voice CapCut / TikTok Free (Tab 3)
- Liệt kê các giọng miễn phí của **CapCut / TikTok** đa ngôn ngữ (Tiếng Việt `BV074_streaming`, `BV075_streaming`, Tiếng Anh, Trung, Nhật, Hàn...) + **300+ giọng miễn phí Edge-TTS Neural** dự phòng tự động.
- Hỗ trợ **Nghe thử giọng** trực tiếp và tạo voice song song có **cache trên ổ đĩa** (chỉnh sửa lại video không cần gọi lại TTS).

### 4. Edit Thuyết minh / Lồng tiếng & Chạy A–Z Hàng Loạt
- **Thuyết minh (`narration`)**: Tự cắt khoảng lặng đầu/cuối mỗi câu, mượn khoảng nghỉ phía sau trước khi tua nhẹ giữ cao độ (`librubberband`), tự động **Auto-Ducking** hạ nhạc/tiếng gốc mượt mà khi giọng đọc cất lên.
- **Lồng tiếng (`dubbing`)**: Cắt khoảng lặng + ép chặt đúng khung thời gian `[start, end]` từng thoại để không đè cảnh/nhân vật, mặc định tắt âm thanh gốc.
- **Chạy A–Z Hàng Loạt (`🚀 Chạy A–Z Hàng Loạt`)**: Chạy gối đầu song song I/O (Dịch + Voice) + AI GPU + Video Encoder (`h264_qsv` / `h264_nvenc`).
