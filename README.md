# Label Tool — Gán nhãn ảnh & huấn luyện YOLO

Ứng dụng máy tính (Python + PySide6/Qt, giao diện kiểu Windows 11 với nền kính trong suốt) để:

1. **Gán nhãn** ảnh bằng khung chữ nhật (bounding box), lưu theo định dạng YOLO.
2. **Xuất dataset** kiểu Roboflow (`train/ valid/ test/` + `data.yaml`), có thể kèm COCO JSON.
3. **Huấn luyện** mô hình YOLO11 bằng [Ultralytics](https://docs.ultralytics.com) ngay trong ứng dụng, rồi xuất `best.pt` sang ONNX.

Công cụ này độc lập với web app Sĩ Số Thông Minh (thư mục `app/`, `server/`), không dùng chung mã.

---

## 1. Cài đặt (Windows)

### Cách nhanh: bấm đúp là chạy

Trong thư mục `label-tool`:

> **Nhận được file `.zip`?** Giải nén trước (chuột phải → *Extract All…*) ra thư mục ngắn, ví dụ `C:\LabelTool`, rồi mới bấm đúp. Bấm đúp ngay bên trong file zip sẽ không chạy được — `install.cmd` sẽ báo và hướng dẫn.

- **`open.cmd`** — bấm đúp để mở ứng dụng. Lần đầu nó tự cài đặt mọi thứ rồi mới mở; những lần sau mở ngay.
- **`install.cmd`** — chỉ cài (hoặc sửa lại) mà không mở ứng dụng.

Việc cài đặt tự làm hết:

1. Tìm Python 3.10–3.14 trên máy. Nếu chưa có, tự tải **Python 3.14** từ python.org và cài cho riêng tài khoản đang dùng (không cần quyền admin).
2. Tạo môi trường ảo `.venv` ngay trong thư mục `label-tool` (không ảnh hưởng phần còn lại của máy). Nếu đường dẫn thư mục quá dài (hơn 70 ký tự) mà Windows chưa bật *Long Path*, `.venv` được đặt ở `%LOCALAPPDATA%\LabelTool\.venv` để tránh giới hạn 260 ký tự của Windows (PySide6 có tệp nằm rất sâu).
3. Cài tất cả thư viện trong `requirements.txt`. Lần đầu mất khoảng 5–15 phút vì PyTorch khá nặng (cần mạng).

Chạy lại bao nhiêu lần cũng được. Khi `requirements.txt` thay đổi, `open.cmd` tự cài thêm phần còn thiếu. Nếu chép thư mục sang máy khác kèm cả `.venv` cũ, nó tự nhận ra `.venv` hỏng và tạo lại.

### Cách thủ công

Cần **Python 3.10 trở lên** (bản cài từ python.org).

```bash
cd label-tool
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Trong Git Bash dùng `source .venv/Scripts/activate` thay cho dòng `activate` ở trên.

- Chỉ cần gán nhãn và xuất dataset? Cài `PySide6-Essentials pillow pyyaml` là đủ; thẻ "Huấn luyện" sẽ báo thiếu thư viện.
- **GPU NVIDIA:** `ultralytics` mặc định kéo về torch bản CPU. Để huấn luyện bằng GPU, cài lại torch bản CUDA theo lệnh mà [pytorch.org](https://pytorch.org/get-started/locally/) đưa ra cho máy bạn, ví dụ:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
  ```

  Thẻ "Huấn luyện" có dòng kiểm tra cho biết GPU có dùng được hay không.

## 2. Chạy

Bấm đúp **`open.cmd`**, hoặc:

```bash
.venv\Scripts\python app.py
```

hoặc từ thư mục gốc của repo: `label-tool\.venv\Scripts\python label-tool\app.py`.

## 3. Cách dùng

### 3.1. Dự án

- **Tạo dự án**: chọn một thư mục trống. Ứng dụng tạo `project.json` và các thư mục con.
- **Mở dự án**: chọn thư mục có `project.json`. Dự án mở gần nhất được tự mở lại lần sau.
- **Nhập ảnh** / **Nhập cả thư mục**: hỗ trợ jpg, png, webp, bmp. Ảnh được *sao chép* vào dự án (ảnh gốc không bị đổi). Ảnh chụp điện thoại bị xoay (EXIF) được xoay đúng chiều khi nhập.

Danh sách bên trái hiện ảnh thu nhỏ và trạng thái: **Chưa gán**, **n khung**, hoặc **Ảnh trống** (đã xem, không có đối tượng). Có bộ lọc "Chưa gán nhãn / Đã gán nhãn".

### 3.2. Gán nhãn

1. Bấm **Thêm** để tạo lớp (ví dụ `nguoi`). Đổi tên/đổi màu bằng nút bên dưới (hoặc nhấp đúp tên lớp để đổi tên).
2. Chọn lớp (bấm vào tên hoặc phím số), rồi **kéo chuột trên ảnh** để vẽ khung.
3. Bấm vào khung để chọn; kéo để di chuyển, kéo các ô vuông nhỏ để đổi kích thước. Muốn đổi lớp một khung: chuyển sang chế độ Chọn/sửa (`V`), chọn khung rồi bấm phím số hoặc tên lớp.

Nhãn được **tự động lưu** (khoảng 0,4 giây sau mỗi thay đổi, và khi chuyển ảnh hoặc đóng ứng dụng).

| Phím | Tác dụng |
|---|---|
| `W` / `V` | Chế độ vẽ khung / chọn – sửa |
| `A`, `←` / `D`, `→` | Ảnh trước / ảnh sau (theo bộ lọc đang chọn) |
| `1`–`9` | Chọn lớp cho khung sắp vẽ; ở chế độ Chọn/sửa (`V`) thì đổi lớp của khung đang chọn |
| `Delete`, `Backspace` | Xoá khung đang chọn |
| `Tab` | Chọn khung kế tiếp |
| `Ctrl+Z` | Hoàn tác |
| `Esc` | Bỏ chọn / huỷ khung đang vẽ |
| `E` | Đánh dấu ảnh đã xong (dùng cho ảnh không có đối tượng) và sang ảnh sau |
| `F`, `0` | Vừa khung nhìn |
| `+` / `-`, lăn chuột | Phóng to / thu nhỏ |
| Chuột phải kéo, hoặc `Space` + kéo | Di chuyển ảnh |
| `H` | Ẩn/hiện tên lớp trên khung |
| `Ctrl+S` | Lưu ngay |

> Ảnh **Chưa gán** sẽ bị bỏ qua khi xuất (trừ khi bật "Gồm cả ảnh chưa gán nhãn"). Ảnh **trống** được xuất như ảnh nền (không có tệp nhãn / tệp rỗng) — giúp mô hình bớt báo nhầm.

### 3.3. Hoàn thành / Xuất dataset

Bấm nút xanh **Hoàn thành / Xuất dataset** ở góc trên bên phải:

- **Tỉ lệ chia** train/valid/test (mặc định 70/20/10) và **seed** (giữ nguyên seed thì lần chia giống nhau).
- **Đổi kích thước**: giữ nguyên; thu nhỏ để cạnh dài nhất bằng W (giữ tỉ lệ, không phóng to ảnh nhỏ); hoặc kéo giãn về đúng W×H. Toạ độ khung được tính lại theo.
- **Tăng cường dữ liệu** (chỉ áp dụng cho tập train): lật ngang, đổi độ sáng ±n%, số bản sao mỗi ảnh.
- **Xuất kèm COCO JSON**: thêm `_annotations.coco.json` trong mỗi tập.

Kết quả (mặc định trong `<dự án>/exports/<tên>-<thời gian>/`):

```
data.yaml
export_info.json
train/images  train/labels
valid/images  valid/labels
test/images   test/labels
```

`data.yaml` không ghi `path`, nên có thể chép cả thư mục sang máy khác vẫn dùng được:

```yaml
train: train/images
val: valid/images
test: test/images
nc: 2
names: [nguoi, ghe]
```

Bảng tóm tắt cho biết số ảnh và số khung của từng lớp trong mỗi tập. Bấm **Dùng để huấn luyện →** để chuyển sang thẻ Huấn luyện với `data.yaml` vừa xuất.

### 3.4. Huấn luyện

Chuyển sang thẻ **Huấn luyện** bằng nút ở góc trên (cạnh "Gán nhãn").


| Tuỳ chọn | Ý nghĩa |
|---|---|
| data.yaml | Dataset vừa xuất (hoặc bất kỳ dataset YOLO/Roboflow nào) |
| Mô hình gốc | `yolo11n.pt` (nhẹ, nhanh), `yolo11s.pt`, `yolo11m.pt`, hoặc tệp `.pt` của bạn. Lần đầu sẽ tự tải về `label-tool/weights/` |
| Epochs | Số vòng huấn luyện (50–100 là điểm khởi đầu hợp lý) |
| imgsz | Kích thước ảnh đầu vào, bội số của 32 (mặc định 640) |
| batch | Số ảnh mỗi lượt; giảm nếu báo thiếu bộ nhớ. `-1` = tự chọn (chỉ GPU) |
| patience | Dừng sớm nếu chừng ấy epoch không cải thiện |
| Thiết bị | Tự động / CPU / GPU (CUDA) |

Bấm **Bắt đầu huấn luyện**: nhật ký, tiến độ từng epoch và chỉ số (mAP50, mAP50-95…) hiện trực tiếp; giao diện không bị treo vì huấn luyện chạy ở tiến trình riêng. **Dừng** để huỷ giữa chừng (nếu đã có `best.pt` thì vẫn dùng được).

Khi xong: kết quả nằm trong `<dự án>/runs/<tên lần chạy>/`, trọng số tốt nhất ở `weights/best.pt`. Bấm **Xuất best.pt → ONNX** để có `best.onnx` cạnh nó.

### 3.5. Chuyển dự án sang máy khác (.zip)

Bấm **Chuyển máy (.zip)** ở thanh trên cùng:

- **Xuất dự án ra tệp .zip…**: đóng gói ảnh, nhãn, tên lớp và màu lớp vào một tệp `.zip` (tên mặc định `<dự án>-<ngày giờ>.zip`, đặt cạnh thư mục dự án). Có thể chọn gồm thêm các dataset đã xuất (`exports/`) và kết quả huấn luyện/trọng số (`runs/`) — hai phần này có thể rất nặng nên mặc định không gồm. Tệp `.zip` phải lưu *ngoài* thư mục dự án.
- **Nhập dự án từ tệp .zip…** (trên máy kia): chọn tệp `.zip`, xem tóm tắt (số ảnh, số lớp, số khung), rồi chọn thư mục để đặt dự án. Dự án được giải nén vào một thư mục con mới và mở ngay để làm tiếp.

An toàn khi nhập:

- Không bao giờ ghi đè: nếu đã có thư mục cùng tên thì thêm hậu tố ` (2)`, ` (3)`…
- Giải nén vào thư mục tạm trước, chỉ đổi tên thành thư mục thật khi mọi thứ hợp lệ; lỗi giữa chừng không để lại rác.
- Từ chối tệp `.zip` không phải dự án Label Tool, hoặc có đường dẫn lạ (`..`, đường dẫn tuyệt đối, liên kết tượng trưng); kiểm tra dung lượng trống trước khi giải nén.

Dự án không lưu đường dẫn tuyệt đối nào, nên mở ở ổ/thư mục khác vẫn đúng. Tệp `.zip` chứa ảnh gốc — đừng đưa lên git hay chia sẻ công khai nếu có ảnh người thật.

### 3.6. Chia việc gán nhãn cho nhiều người rồi gộp lại

1. Thống nhất **tên lớp** trước (ví dụ ai cũng dùng `nguoi`, `ghe`). Thứ tự lớp và chữ hoa/thường không quan trọng.
2. Mỗi người nhận một phần ảnh (tạo dự án riêng và nhập phần ảnh của mình, hoặc nhập từ một tệp `.zip` chung), gán nhãn xong thì **Chuyển máy (.zip) → Xuất dự án ra tệp .zip…** rồi gửi tệp cho người tổng hợp.
3. Người tổng hợp bấm **Chuyển máy (.zip) → Gộp nhiều tệp .zip thành một dự án…**, chọn cùng lúc tất cả các tệp `.zip` (giữ Ctrl để chọn nhiều), rồi chọn **gộp vào dự án đang mở** hoặc **tạo dự án mới** (chọn một thư mục trống).
4. Xong sẽ có bảng tóm tắt từng tệp: số ảnh thêm vào, ảnh trùng, lớp mới. Sau đó xuất dataset và huấn luyện như bình thường.

Quy tắc gộp:

- **Lớp ghép theo tên**: `Nguoi` và `nguoi` là một lớp; lớp chưa có được thêm vào cuối danh sách.
- **Ảnh trùng nội dung** (cùng một ảnh, kể cả khác tên tệp) chỉ giữ một bản. Nếu dự án đích chưa gán nhãn ảnh đó thì lấy nhãn từ tệp `.zip`.
- **Hai người gán khác nhau cho cùng một ảnh**: giữ nhãn đang có (của tệp gộp trước), và tên ảnh được liệt kê trong bảng tóm tắt để bạn mở ra xem lại.
- **Ảnh khác nhau nhưng trùng tên**: ảnh sau được đổi tên (`_1`, `_2`…), không ghi đè.
- Không xoá hay sửa dữ liệu đang có. Gộp lại cùng một tệp `.zip` lần nữa sẽ không thêm gì.

### 3.7. Dùng mô hình trong web app

Web app nạp mô hình ở `public/models/yolo11n.onnx`, ảnh vào 640×640 và chỉ đếm lớp **person (chỉ số 0)**. Nếu thay bằng mô hình tự huấn luyện:

- Huấn luyện và xuất ONNX với **imgsz 640**.
- Dùng họ **YOLO11** (công cụ cố ý không đưa YOLO26 vào vì đầu ra của nó khác, web app không đọc được).
- Đặt lớp người là **lớp đầu tiên** (chỉ số 0) trong danh sách lớp, hoặc sửa web app cho khớp.

## 4. Cấu trúc thư mục dự án

```
<dự án>/
  project.json     tên dự án, danh sách lớp và màu
  images/          ảnh đã nhập
  labels/          nhãn YOLO: <tên ảnh>.txt, mỗi dòng "lớp xc yc w h" (chuẩn hoá 0–1)
  exports/         các dataset đã xuất
  runs/            kết quả huấn luyện
```

Mã nguồn:

```
label-tool/
  app.py                 điểm chạy chính
  labeltool/project.py   dự án, đọc/ghi nhãn YOLO
  labeltool/exporter.py  chia tập, tăng cường, data.yaml, COCO
  labeltool/archive.py   xuất/nhập cả dự án dưới dạng .zip (chuyển máy)
  labeltool/merge.py     gộp nhiều tệp .zip (nhiều người gán nhãn) thành một dự án
  labeltool/train_worker.py / train_runner.py   huấn luyện ở tiến trình riêng
  labeltool/ui/          giao diện Qt: main_window (cửa sổ chính), annotator (khung vẽ),
                         thumb_list, export_dialog, zip_dialog, train_tab, theme (màu/QSS),
                         glass (nền kính), icons
  tools/make_synthetic.py  tạo ảnh tổng hợp (hình tròn/vuông) để thử
  tests/test_core.py       kiểm thử phần lõi
```

## 5. Thử nhanh với dữ liệu tổng hợp

Không cần ảnh thật:

```bash
.venv\Scripts\python tools\make_synthetic.py D:\thu-nghiem --count 40 --project
.venv\Scripts\python app.py
```

Mở dự án `D:\thu-nghiem` (đã có sẵn nhãn hình tròn/hình vuông), xuất dataset rồi huấn luyện 1–2 epoch trên CPU để kiểm tra.

Kiểm thử phần lõi:

```bash
.venv\Scripts\python tests\test_core.py
```

## 6. Lưu ý

- `.venv/`, `weights/`, `runs/`, `exports/`, tệp `.pt`, `.onnx` và `.zip` trong `label-tool/` đã được bỏ qua trong `.gitignore`. **Không đưa ảnh có người thật, dataset hay trọng số lên git.**
- Cài đặt nhỏ (dự án mở gần nhất, kiểu nền, độ đục) lưu ở `%USERPROFILE%\.labeltool\settings.json`.
- **Nền cửa sổ** (góc dưới bên phải): *Kính trong (Acrylic)* giống File Explorer, *Kính mờ (Mica)*, hoặc *Nền đặc*. Hiệu ứng kính cần Windows 11 bản 22H2 trở lên; máy khác tự dùng nền đặc. Ảnh đang gán nhãn luôn hiển thị đặc, không bị nhoè.
- **Độ đục** (thanh trượt cạnh ô Nền cửa sổ, 0–90%, mặc định 20%): kéo về trái để kính trong hơn, kéo sang phải nếu chữ khó đọc trên hình nền sáng. Chỉ có tác dụng khi dùng nền kính.
- Biểu tượng nút lấy từ phông Segoe Fluent Icons / Segoe MDL2 Assets có sẵn trên Windows.
