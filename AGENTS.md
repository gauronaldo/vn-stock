# AGENTS.md — vn-factor-ranking

Đặc tả để một coding agent thực hiện dự án từ đầu đến cuối. Đọc hết file trước khi viết code.

- **Tên repo:** `vn-factor-ranking` · **Tên package:** `vnfactor`
- **Chủ dự án:** sinh viên ứng tuyển Data Science Intern; repo là project showcase công khai trên GitHub.
- **Ngôn ngữ trong repo:** code, docstring, commit message, README và báo cáo viết bằng tiếng Anh. Trao đổi với chủ dự án bằng tiếng Việt.
- **Môi trường:** Windows 11, PowerShell, Python 3.12. Mọi lệnh hướng dẫn phải chạy được trên PowerShell.
- **Phạm vi:** file này mô tả **bản 1 (bản nộp đơn)**. Những thứ ở mục 11 "Để sau" không được làm khi chưa có yêu cầu của chủ dự án, kể cả khi thấy "nên có".

---

## 1. Mục tiêu

Xây một pipeline tái lập được: vào ngày giao dịch cuối mỗi tháng, xếp hạng các cổ phiếu trong rổ thanh khoản cao của HOSE theo điểm nhân tố, rồi đánh giá trung thực xem thứ hạng đó có dự báo được lợi suất tháng kế tiếp hay không.

**Câu hỏi nghiên cứu**

1. Từng nhân tố có sức dự báo cắt ngang trên thị trường Việt Nam không, đo bằng rank IC?
2. Điểm tổng hợp đẳng trọng số và mô hình học máy (Ridge, LightGBM) có tốt hơn từng nhân tố riêng lẻ không? Mô hình học máy có thắng được cách cộng điểm đơn giản không?
3. Sau chi phí giao dịch, danh mục nhóm đầu có còn vượt danh mục đẳng trọng số của cả rổ không?

**Sản phẩm bàn giao**

- Package `vnfactor` chạy được từng bước bằng `python -m vnfactor <bước>`.
- Bộ test chạy trên dữ liệu tổng hợp, CI chạy ruff và pytest.
- `reports/report.md` kèm hình và bảng thống kê tổng hợp.
- `README.md` để người tuyển dụng đọc 3 phút hiểu được bài toán, phương pháp, kết quả và giới hạn.

**Tiêu chí thành công là chất lượng phương pháp, không phải lợi suất.** Kết quả "nhân tố X không có sức dự báo" hay "LightGBM thua cách cộng điểm" là kết quả hợp lệ và phải báo cáo nguyên trạng.

---

## 2. Quy tắc cứng

Khi một yêu cầu khác mâu thuẫn với các quy tắc này, các quy tắc này thắng.

1. **Không nhìn trước tương lai.** Tín hiệu tại ngày `t` chỉ dùng giá có ngày ≤ `t` và báo cáo tài chính có `available_date` ≤ `t`. Mô hình học máy chỉ huấn luyện trên nhãn đã biết tại thời điểm dự báo. Có test chứng minh (mục 8).
2. **Không lộ khoá.** Không ghi khoá API vào code, log, commit hay tin nhắn. Không đọc hay in nội dung `.env`. Không yêu cầu chủ dự án dán khoá vào chat; khi cần khoá, nhờ họ tự chạy `vnstock keys set <nguồn>` trong terminal.
3. **Không commit dữ liệu thị trường.** Toàn bộ `data/` nằm trong `.gitignore`. Điều khoản của nguồn không cho phép phân phối lại dữ liệu gốc lẫn bảng đã chuẩn hoá. Repo chỉ chứa code, hình và bảng thống kê tổng hợp.
4. **Chỉ `src/vnfactor/client.py` được import `vnstock`.** Phần còn lại làm việc với schema nội bộ ở mục 7.2.
5. **Không tin trí nhớ về vnstock.** Các lớp `Vnstock().stock(...)`, `Quote`, `Listing`, `Finance` và nguồn `VCI`/`TCBS`/`KBS` thuộc các thế hệ đã ngừng hoạt động. Tra chữ ký thật bằng `vnstock tools --source <nguồn>`, `show_api()`, `show_doc()` hoặc tài liệu ở mục 3.4.
6. **Holdout chỉ chạy một lần**, sau khi mọi lựa chọn thiết kế và siêu tham số đã chốt trong `docs/methodology.md`.
7. **Lỗi dữ liệu phải nổi lên.** Không bắt exception rồi thay bằng bảng rỗng. Không điền dữ liệu thiếu bằng 0.
8. **Tôn trọng giới hạn của nguồn.** Tải tuần tự, có giãn cách, có cache, sau 15:00. Không chạy song song, không lách giới hạn tần suất.
9. **Báo cáo trung thực.** Test fail thì nói fail. Bước bị bỏ qua hay giả định chưa kiểm chứng thì ghi vào `docs/methodology.md` và phần Limitations của README.

Khi thực tế khác với giả định trong file này và file không nêu phương án dự phòng, hỏi chủ dự án trước khi tự quyết.

---

## 3. Dữ liệu: vnstock thế hệ 6

Thông tin dưới đây lấy từ tài liệu chính thức (đối chiếu ngày 12/11/2026) và đã kiểm chứng bằng lời gọi thật với nguồn `fhsc` ngày 13/11/2026 (`docs/data_audit.md`). Các lời gọi `ssi` chưa kiểm chứng vì chưa có khoá.

### 3.1. Bối cảnh

- **Thế hệ 5 và cũ hơn đã ngừng hoạt động từ 10/11/2026.** Dự án dùng `vnstock` 6.x, cài từ kho của vnstocks.com, không có trên PyPI.
- **Mô hình "tự mang khoá".** Dữ liệu đến từ Open API của công ty chứng khoán bằng tài khoản của chủ dự án. Không có nguồn mặc định: mỗi lời gọi phải chỉ rõ `source=`.
- **Không dùng `vnstock_quant` và `vnstock_pipeline`** (bản có phí). Tự tính mọi chỉ số bằng pandas.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -U --extra-index-url https://vnstocks.com/api/packages "vnstock==6.0.0" "vnstock-oapi==0.1.1"
vnstock doctor
```

`client.py` import `vnstock` bên trong hàm, để test và CI chạy được khi chưa cài.

### 3.2. Nguồn và khoá

| `source=` | Dùng cho | Cần | Lưu ý |
| :-- | :-- | :-- | :-- |
| `fhsc` | **Nguồn duy nhất của bản 1:** danh sách mã, nến giá, giá trị khớp lệnh, số cổ phiếu lưu hành, báo cáo tài chính | API Key | Chỉ phục vụ 06:00–21:00 giờ Việt Nam. Nến ngày có từ 03/01/2017. `volume_history` có từ 02/08/2021, mỗi lời gọi tối đa 5 năm |
| `ssi` | **Không dùng ở bản 1** (chưa có khoá). Có `trade_history`, có thể có nến sâu hơn 2017 | API Key + Secret | Nếu muốn thêm thì phải quyết định trước khi chạy holdout |

`fhsc` là bắt buộc và là nguồn duy nhất của bản 1. Agent chỉ được chạy `vnstock keys` và `vnstock keys check` để xác nhận trạng thái khoá.

### 3.3. Lời gọi dự kiến

```python
from vnstock import Market, Reference, Fundamental

Reference("fhsc").index("VNINDEX").constituents()  # 409 mã HOSE HIỆN TẠI, không có lịch sử
Reference("fhsc").company("FPT").listing()  # outstanding_shares hiện tại
Market("fhsc").equity("FPT").ohlcv(start="2017-01-01", end="2026-10-30", interval="1D")
Market("fhsc").equity("FPT").volume_history(start="2021-11-01", end="2026-10-30")  # tối đa 5 năm
Fundamental("fhsc").equity("FPT").income_statement(period="quarter", format="long")
Fundamental("fhsc").equity("FPT").balance_sheet(period="quarter", format="long")
```

- `ohlcv()` trả `time, open, high, low, close, volume`; giá bằng **nghìn đồng**, **đã điều chỉnh** theo sự kiện quyền; khối lượng là số gốc, không điều chỉnh. Nếu `df.attrs["last_bar_partial"]` là `True` thì bỏ nến cuối.
- Truyền `compat="v6"` ở mọi lời gọi (mặc định là tên cột kiểu thế hệ 5).
- Báo cáo tài chính: số quý là số riêng của quý, đơn vị đồng; `BS_EQUITY` gồm cả lợi ích cổ đông không kiểm soát; loại hình doanh nghiệp (`regular`, `bank`, `securities`, `insurance`) nằm ở `df.attrs["company_type"]`. Cột giá trị là `item_value`. Nhãn kỳ dạng `2025-Q4`, theo quý dương lịch kể cả với mã có năm tài chính lệch (đã kiểm chứng với `HSG`). Không có ngày công bố.
- Không có Gói quyền sử dụng thì chỉ nhận **bộ chỉ tiêu rút gọn** (chỉ tiêu cấp 1). Bộ này có lợi nhuận sau thuế của cổ đông công ty mẹ và `BS_EQUITY`, nhưng không có vốn góp của chủ sở hữu; lợi ích cổ đông không kiểm soát trên bảng cân đối chỉ có ở mẫu ngân hàng.
- Lỗi có kiểu (`RateLimitError`, `NetworkError`, `CredentialError`, `SourceError`...). Chỉ thử lại có backoff với `RateLimitError` và `NetworkError`.

### 3.4. Tài liệu

- https://vnstocks.com/docs/vnstock/v6 (tổng quan, từ đó dẫn tới Reference, Market, Fundamental)
- https://vnstocks.com/docs/vnstock/v6/cai-dat
- https://vnstocks.com/docs/vnstock/v6/tham-khao/nguon-du-lieu (phạm vi nguồn và điều khoản)

---

## 4. Kiểm toán dữ liệu (cổng kiểm tra)

Trước khi viết pipeline, trả lời bốn câu hỏi bằng lời gọi thật trên `FPT`, `HPG`, `VCB`, `SSI`, `BVH`, `HSG`. Ghi câu trả lời và bằng chứng (số dòng, khoảng ngày, tên cột) vào `docs/data_audit.md`, không ghi dữ liệu thô.

| # | Câu hỏi | Quyết định |
| :-- | :-- | :-- |
| A1 | Nến ngày sâu tới năm nào? Một lời gọi lấy được bao nhiêu năm? | Ngày bắt đầu backtest. Dưới 5 năm thì dừng và báo chủ dự án |
| A2 | Có lấy được giá đóng cửa chưa điều chỉnh và giá trị khớp lệnh lịch sử không? Ứng viên: `Reference("ssi").company(mã).trading_stats()`, `Market(...).equity(mã).trade_history()` | Công thức vốn hoá (mục 5.4) |
| A3 | Bộ chỉ tiêu rút gọn có những `item_id` nào ở cả bốn mẫu biểu? Cần: vốn chủ sở hữu, lợi ích cổ đông không kiểm soát, lợi nhuận sau thuế của cổ đông công ty mẹ, vốn góp của chủ sở hữu | `ep`, `bp`, `roe` có tính được không |
| A4 | Báo cáo quý sâu tới năm nào? Cộng 4 quý có khớp số năm không? | Ngày bắt đầu của nhân tố cơ bản; cách tính TTM |

Nếu trong lúc kiểm tra tình cờ thấy điều bất thường (có cột ngày công bố, nhãn kỳ lạ ở doanh nghiệp có năm tài chính lệch như `HSG`), ghi lại một dòng, không điều tra thêm.

**Cổng kiểm tra:** xong bốn câu thì dừng, tóm tắt cho chủ dự án ngày bắt đầu backtest, mốc holdout và công thức vốn hoá được chọn. Chỉ tiếp tục khi chủ dự án đồng ý.

**Kết quả (chủ dự án duyệt ngày 13/11/2026, bằng chứng ở `docs/data_audit.md`):**

- Chỉ dùng nguồn `fhsc`. Các ứng viên `ssi` ở A2 chưa kiểm tra.
- A1: nến ngày có từ 03/01/2017. Rebalance đầu 31/01/2018, cuối 31/08/2026, tổng 104 tháng.
- Giai đoạn nghiên cứu: 73 tháng, tới 31/01/2024. Holdout: 31 tháng, từ 29/02/2024.
- A2: không có giá chưa điều chỉnh. Giá trị khớp lệnh thật chỉ có từ 02/08/2021, nên dùng giá trị giao dịch có neo (mục 5.2).
- A3: vốn hoá theo phương án 2. `bp` và `roe` dùng vốn chủ sở hữu tổng (mục 5.4).
- A4: báo cáo quý có từ 2009–2013 tùy mã, không thiếu quý. TTM là tổng 4 quý. Nhãn quý theo quý dương lịch.

---

## 5. Phương pháp

Mọi tham số nằm ở `configs/base.yaml`, không hardcode.

### 5.1. Quy ước thời gian

- **Ngày tái cân bằng `t`:** ngày giao dịch cuối cùng của mỗi tháng. Tín hiệu dùng dữ liệu tới hết phiên `t`.
- **Thực thi:** giá đóng cửa phiên `t+1`.
- **Lợi suất kỳ tới:** từ giá đóng cửa `t+1` tới giá đóng cửa phiên ngay sau ngày tái cân bằng kế tiếp. Dùng chung cho IC, nhãn học máy và backtest.
- **Chia giai đoạn:** 70% số tháng đầu là giai đoạn nghiên cứu, 30% cuối là holdout. Chốt mốc ngày vào config ngay sau kiểm toán dữ liệu. Đã chốt: rebalance đầu 31/01/2018, cuối 31/08/2026 (104 tháng); nghiên cứu tới 31/01/2024 (73 tháng), holdout từ 29/02/2024 (31 tháng).

### 5.2. Rổ cổ phiếu

Không dùng danh sách VN30 hay VN100 hiện tại để backtest quá khứ: nguồn chỉ trả thành phần hôm nay, dùng nó là đưa thiên lệch sống sót vào kết quả. Dựng rổ theo quy tắc tại mỗi `t`:

1. Ứng viên: cổ phiếu thường trên HOSE, lấy từ thành phần VNINDEX hiện tại của `fhsc`.
2. Giữ mã có ít nhất 252 phiên lịch sử và có giao dịch ở ít nhất 80% số phiên trong 126 phiên gần nhất.
3. Lấy 100 mã có trung vị giá trị giao dịch ngày 126 phiên cao nhất.

**Giá trị giao dịch ngày** (cột `value`, dựng ở bước ingest, không dựng trong `universe.py`):

- Từ 02/08/2021: `matched_value` của nguồn.
- Trước đó: `close_adj × 1000 × volume / k`. `k` của từng mã là trung vị của `close_adj × 1000 × volume / matched_value` trong 20 phiên đầu tiên có `matched_value`.
- Cách này gỡ phần điều chỉnh giá phát sinh sau mốc neo. Các sự kiện quyền trong 2017–2021 chưa gỡ được; ghi vào Limitations.

Mã đã huỷ niêm yết không được kiểm tra ở bản 1; ghi vào Limitations rằng thiên lệch sống sót vẫn còn một phần.

Nguồn trả toàn bộ lịch sử của mã, không có sàn theo thời gian hay ngày chuyển sàn, nên mã chuyển từ HNX hoặc UPCoM lên HOSE mang theo cả giai đoạn trước khi lên HOSE. Chủ dự án quyết định ngày 14/11/2026: giữ nguyên giai đoạn này và ghi vào Limitations (bằng chứng ở `docs/data_audit.md`, mục Ingest findings).

### 5.3. Báo cáo tài chính theo thời điểm

- `period_end` = ngày cuối quý. `available_date = period_end + 45 ngày` (hạn công bố báo cáo quý là 20–30 ngày và thường có gia hạn).
- Tại `t`, chỉ dùng các kỳ có `available_date ≤ t`, nối bằng as-of join.
- **TTM** = tổng 4 quý liên tiếp gần nhất đã khả dụng. Thiếu một quý thì NaN.
- Số của nguồn có thể đã điều chỉnh hồi tố; không khắc phục được, ghi vào Limitations.

### 5.4. Nhân tố

Cột "dấu" là giả thuyết đặt trước khi xem dữ liệu. Dấu dương: giá trị cao kỳ vọng đi kèm lợi suất kỳ tới cao.

| Nhóm | Nhân tố | Định nghĩa tại `t` | Dấu |
| :-- | :-- | :-- | :-- |
| Momentum | `mom_12_1` | `P[t-21] / P[t-252] - 1` | + |
| Đảo chiều | `rev_1m` | `P[t] / P[t-21] - 1` | − |
| Biến động | `vol_126` | Độ lệch chuẩn log-return ngày trong 126 phiên | − |
| Thanh khoản | `adv_63` | Log trung vị giá trị giao dịch ngày trong 63 phiên | − |
| Giá trị | `ep` | Lợi nhuận sau thuế cổ đông công ty mẹ TTM / vốn hoá | + |
| Giá trị | `bp` | Vốn chủ sở hữu tổng (`BS_EQUITY`, gồm cả lợi ích cổ đông không kiểm soát) / vốn hoá | + |
| Chất lượng | `roe` | Lợi nhuận sau thuế tổng TTM / vốn chủ sở hữu tổng bình quân | + |

`P` là giá đóng cửa đã điều chỉnh, chỉ số theo số phiên giao dịch. Cửa sổ thiếu quá 20% số phiên thì nhân tố là NaN. Vốn chủ sở hữu ≤ 0 thì `bp` và `roe` là NaN.

**Dùng E/P và B/P, không dùng P/E và P/B:** P/E vô nghĩa khi lợi nhuận âm và không xếp hạng được quanh 0.

**Vốn hoá tại `t`**: đã chọn **phương án 2** theo kết quả kiểm toán. Phương án 1 không làm được vì nguồn không có giá chưa điều chỉnh trước 2021 và không có vốn góp.

- *Phương án 1:* giá chưa điều chỉnh × (vốn góp của chủ sở hữu ở kỳ đã khả dụng gần nhất / 10.000 đồng).
- *Phương án 2 (đang dùng):* giá đã điều chỉnh × số cổ phiếu lưu hành hiện tại. Đúng với cổ tức bằng cổ phiếu và chia tách. Sai lệch khi có cổ tức tiền mặt (vốn hoá quá khứ thấp đi) và phát hành thêm (cao lên), nên `ep` và `bp` chứa thông tin tương lai ở tầng dữ liệu; ghi vào Limitations.

### 5.5. Tiền xử lý và điểm tổng hợp

Tại mỗi `t`, trên các mã trong rổ:

1. Winsorize ở phân vị 1% và 99%, rồi z-score.
2. `ep`, `bp`, `roe` được z-score **trong từng nhóm**: tài chính (`bank`, `securities`, `insurance`) và phi tài chính. Bảng cân đối của ngân hàng khác hẳn doanh nghiệp sản xuất, so thẳng là sai.
3. Nhân với dấu giả thuyết. Không điền giá trị thiếu.

**Điểm tổng hợp** = trung bình đẳng trọng số của bốn nhóm: momentum (`mom_12_1`), biến động (`vol_126`), giá trị (trung bình `ep`, `bp`), chất lượng (`roe`). Mã có ít hơn 2 nhóm thì không xếp hạng. `rev_1m` và `adv_63` chỉ báo cáo như nhân tố đơn và làm đặc trưng cho mô hình học máy. Trọng số ghi vào `docs/methodology.md` trước khi đánh giá, không tối ưu theo kết quả.

### 5.6. Đánh giá

Cho từng nhân tố, điểm tổng hợp và từng mô hình học máy:

- Rank IC (Spearman) theo tháng: trung bình, ICIR, t-stat, tỷ lệ tháng IC > 0.
- Lợi suất trung bình theo ngũ phân vị và chênh lệch Q5 − Q1.

Báo cáo nêu một câu rằng thử 7 nhân tố thì vài t-stat > 2 có thể do may rủi.

### 5.7. Backtest

- **Danh mục:** mua nhóm ngũ phân vị cao nhất, đẳng trọng số, tái cân bằng hằng tháng, chỉ mua (ở Việt Nam không bán khống được).
- **Chuẩn so sánh:** danh mục đẳng trọng số của cả rổ. Q5 − Q1 chỉ là thước đo tín hiệu trên giấy.
- **Chi phí:** 0,15% mỗi chiều cộng 0,1% thuế trên giá trị bán, tính trên phần danh mục thay đổi. Báo cáo trước và sau phí.
- **Chỉ số:** CAGR, độ biến động, Sharpe, sụt giảm tối đa, lợi suất vượt chuẩn, tỷ lệ luân chuyển.

Tự viết bằng pandas dạng ma trận trọng số × lợi suất, không dùng framework backtest.

### 5.8. Mô hình học máy (bắt buộc)

- **Đặc trưng:** z-score của cả 7 nhân tố. **Nhãn:** hạng phần trăm của lợi suất kỳ tới trong từng tháng.
- **Mô hình:** Ridge và LightGBM, mỗi mô hình một bộ siêu tham số nhỏ, cố định, ghi vào config trước khi chạy. Không dò siêu tham số.
- **Walk-forward cửa sổ mở rộng:** cần tối thiểu 36 tháng huấn luyện, huấn luyện lại mỗi 12 tháng. Khi dự báo cho tháng `t`, tập huấn luyện chỉ gồm các tháng có nhãn đã biết tại `t`, tức bỏ tháng liền trước.
- **Giá trị thiếu:** LightGBM nhận NaN trực tiếp. Ridge thay NaN bằng 0 (trung bình cắt ngang của z-score); đây là ngoại lệ duy nhất của quy tắc 7, ghi rõ trong báo cáo.
- **So sánh:** cùng các tháng ngoài mẫu, đặt cạnh điểm tổng hợp đẳng trọng số: IC và lợi suất nhóm đầu sau phí. Mẫu chỉ cỡ 100 mã × hơn 100 tháng nên mô hình phức tạp có thể thua; nếu thua thì báo cáo là thua và giải thích vì sao.

### 5.9. Độ vững

Chạy trên giai đoạn nghiên cứu, mỗi lần đổi một yếu tố:

- `lag_days` 45 so với 90.
- Chi phí gấp đôi.
- **Minh hoạ thiên lệch sống sót:** chạy lại với rổ "thành phần VN100 hiện tại áp ngược về quá khứ", so với rổ dựng theo quy tắc. `fhsc` không có VN100 (chỉ có VN30 và VNINDEX); rổ thay thế chờ chủ dự án quyết định trước giai đoạn 7.

---

## 6. Cấu trúc repo

```
vn-factor-ranking/
├── AGENTS.md
├── README.md
├── LICENSE                  # MIT, chỉ áp dụng cho code
├── pyproject.toml
├── .gitignore               # data/, .env, .venv/
├── .env.example             # chỉ tên biến
├── .github/workflows/ci.yml
├── configs/base.yaml
├── data/                    # KHÔNG commit
│   ├── raw/                 # mỗi mã một file, đã đổi về schema nội bộ
│   └── processed/           # universe, factors, scores, returns, predictions
├── docs/
│   ├── data_audit.md
│   └── methodology.md       # định nghĩa, giả thuyết, tham số chốt trước, nhật ký quyết định
├── scripts/
│   └── data_audit.py        # chạy tay một lần ở giai đoạn 1
├── reports/
│   ├── report.md
│   ├── figures/
│   └── tables/              # chỉ thống kê tổng hợp
├── src/vnfactor/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py               # argparse, mỗi bước một lệnh con
│   ├── config.py            # đọc YAML thành dict, kiểm tra tối thiểu
│   ├── client.py            # nơi DUY NHẤT import vnstock
│   ├── ingest.py            # tải, giãn cách, thử lại, cache parquet
│   ├── calendar.py          # ngày tái cân bằng, lợi suất kỳ tới
│   ├── universe.py
│   ├── fundamentals.py      # available_date, TTM, as-of join
│   ├── factors.py           # 7 nhân tố, tiền xử lý, điểm tổng hợp
│   ├── evaluation.py        # IC, ngũ phân vị
│   ├── backtest.py          # danh mục, chi phí, chỉ số
│   ├── ml.py                # walk-forward Ridge và LightGBM
│   └── report.py            # hình, bảng, report.md
└── tests/
    ├── conftest.py          # sinh dữ liệu tổng hợp nhỏ cho test
    ├── test_config.py
    ├── test_cli.py
    ├── test_client.py
    ├── test_universe.py
    ├── test_fundamentals.py
    ├── test_factors.py
    ├── test_no_lookahead.py
    ├── test_evaluation.py
    ├── test_backtest.py
    └── test_ml.py
```

Logic nằm trong `src/`. Hàm tính toán là hàm thuần (DataFrame vào, DataFrame ra); đọc ghi file chỉ ở `ingest.py` và `cli.py`. Chỉ tạo file khi tới giai đoạn cần nó, không tạo file rỗng giữ chỗ.

---

## 7. Chuẩn kỹ thuật

### 7.1. Công cụ và lệnh

- Thư viện: `pandas`, `numpy`, `pyarrow`, `scipy`, `matplotlib`, `pyyaml`, `scikit-learn`, `lightgbm`. Dev: `pytest`, `ruff`.
- `ruff` cho lint và format. Type hint ở hàm public. Docstring ngắn, nêu đơn vị và quy ước thời gian. Seed cố định.

```powershell
python -m vnfactor ingest        # tải nến và báo cáo tài chính, tiếp tục được khi bị ngắt
python -m vnfactor build         # rổ, nhân tố, điểm, lợi suất kỳ tới
python -m vnfactor evaluate      # IC, ngũ phân vị, backtest trên giai đoạn nghiên cứu
python -m vnfactor ml            # walk-forward trên giai đoạn nghiên cứu
python -m vnfactor robustness
python -m vnfactor holdout       # chỉ chạy một lần, cuối dự án
python -m vnfactor report
```

Mỗi lệnh đọc `configs/base.yaml`, in số dòng vào và ra, thoát với mã khác 0 khi lỗi.

### 7.2. Schema nội bộ (parquet)

| Bảng | Khoá | Cột |
| :-- | :-- | :-- |
| `prices_daily` | `date, symbol` | `open, high, low, close_adj, volume, value, value_source` (`value_source` là `matched` hoặc `proxy`) |
| `fundamentals_q` | `symbol, report_period, item_id` | `period_end, available_date, value, company_type` |
| `shares_current` | `symbol` | `outstanding_shares` (giá trị hiện tại, không có lịch sử) |
| `universe` | `rebalance_date, symbol` | `adv_126, sector_group` |
| `factors` | `rebalance_date, symbol, factor` | `raw, z` |
| `scores` | `rebalance_date, symbol, model` | `score, quintile` |
| `fwd_returns` | `rebalance_date, symbol` | `fwd_ret_1m` |

Giá bằng nghìn đồng, giá trị giao dịch bằng đồng, ngày không kèm múi giờ. `item_id` nội bộ: `equity` (tổng, gồm lợi ích cổ đông không kiểm soát), `npat_parent`, `npat_total`; ánh xạ từ mã của vnstock nằm trong `client.py`. `model` nhận `composite`, `ridge`, `lightgbm`.

### 7.3. Ingest

- Trình tự: thành phần VNINDEX → nến ngày → `volume_history` theo đoạn 5 năm (từ 02/08/2021) → dựng cột `value` → tính ADV → tải báo cáo tài chính và số cổ phiếu lưu hành cho các mã từng lọt top 150 theo ADV.
- Mỗi mã một file; đã có thì bỏ qua trừ khi truyền `--refresh`.
- Giãn cách lấy từ config (mặc định 1 giây). Backoff luỹ thừa khi gặp `RateLimitError`, tối đa 5 lần rồi dừng.
- Sau ingest in ra: số mã, khoảng ngày, tỷ lệ thiếu, các mã có lợi suất ngày vượt biên độ ±7%.

### 7.4. Git

Mỗi giai đoạn ít nhất một commit, message dạng Conventional Commits. Trước mỗi commit chạy `ruff check .`, `ruff format --check .`, `pytest`. Không commit hay push khi chủ dự án chưa yêu cầu.

---

## 8. Kiểm thử bắt buộc

Chạy trên dữ liệu tổng hợp sinh trong `tests/conftest.py`, không gọi mạng.

1. **Không nhìn trước:** tính nhân tố và điểm tại `t` trên dữ liệu đầy đủ, rồi trên dữ liệu đã cắt bỏ mọi thứ sau `t`. Hai kết quả phải trùng khớp, với ít nhất 5 ngày `t`. Đây là test quan trọng nhất của repo.
2. **As-of join:** báo cáo có `available_date = t + 1 ngày` không xuất hiện trong tín hiệu ngày `t`; `available_date = t` thì có.
3. **TTM:** đủ 4 quý liên tiếp thì bằng tổng; thiếu một quý thì NaN.
4. **Công thức nhân tố:** mỗi nhân tố một ví dụ nhỏ tính tay.
5. **Tiền xử lý:** sau z-score, trung bình ≈ 0 và độ lệch chuẩn ≈ 1 trong từng nhóm; NaN vẫn là NaN.
6. **Rổ:** mã chưa đủ lịch sử bị loại; không quá `top_n` mã; rổ tại `t` không đổi khi thêm dữ liệu sau `t`.
7. **Backtest:** trọng số cộng lại bằng 1; danh mục nắm cả rổ với phí 0 trùng chuẩn so sánh; phí làm giảm lợi suất đúng bằng luân chuyển × mức phí.
8. **IC:** nhân tố xếp đúng thứ tự lợi suất thì IC = 1, xếp ngược thì IC = −1.
9. **Walk-forward:** với mọi tháng dự báo `t`, tập huấn luyện không chứa tháng nào có nhãn chưa biết tại `t`.
10. **Client:** mock `vnstock`; kiểm tra đổi tên cột về schema nội bộ, bỏ nến dở dang, lỗi của nguồn được ném lại.

CI: Ubuntu, Python 3.12, cài `.[dev]`, chạy ruff và pytest. Không cài `vnstock`, không có secret.

---

## 9. Lộ trình

Làm lần lượt. Mỗi giai đoạn kết thúc bằng một báo cáo ngắn cho chủ dự án.

| # | Giai đoạn | Việc chính | Hoàn thành khi |
| :-- | :-- | :-- | :-- |
| 0 | Khung repo | `pyproject.toml`, config, CLI tối giản, CI, fixture dữ liệu tổng hợp | ruff và pytest xanh |
| 1 | Kiểm toán dữ liệu | `client.py` tối thiểu, `scripts/data_audit.py`, trả lời A1–A4 | `docs/data_audit.md` xong. **Dừng, chờ duyệt** |
| 2 | Ingest | `client.py`, `ingest.py` | Tải xong; chạy lại không gọi mạng; test 10 xanh |
| 3 | Rổ và dữ liệu theo thời điểm | `calendar.py`, `universe.py`, `fundamentals.py` | Test 2, 3, 6 xanh |
| 4 | Nhân tố và điểm tổng hợp | `factors.py`; chốt `docs/methodology.md` | Test 1, 4, 5 xanh |
| 5 | Đánh giá và backtest | `evaluation.py`, `backtest.py` | Test 7, 8 xanh; bảng IC và đường vốn giai đoạn nghiên cứu |
| 6 | Học máy | `ml.py` | Test 9 xanh; bảng so sánh với điểm tổng hợp |
| 7 | Độ vững | Ba phép thử ở mục 5.9 | Bảng độ vững |
| 8 | Holdout | Chạy `holdout` đúng một lần | Kết quả holdout nằm cạnh giai đoạn nghiên cứu, không sửa gì sau đó |
| 9 | Báo cáo | `report.py`, `report.md`, README | Đạt mục 10 |

Nếu thiếu thời gian: bỏ `rev_1m` và `adv_63` trước, rồi tới hai phép thử độ vững về độ trễ và chi phí. Không bỏ kiểm toán dữ liệu, test không nhìn trước, học máy hay holdout.

---

## 10. README, báo cáo và định nghĩa "xong"

`README.md` theo thứ tự:

1. Một đoạn: bài toán, dữ liệu, kết quả chính kèm con số trên holdout, nói thẳng cả khi kết quả yếu.
2. Một hình: đường vốn nhóm đầu so với chuẩn, hoặc lợi suất tích luỹ theo ngũ phân vị.
3. Phương pháp tóm tắt và bảng IC (nhân tố, điểm tổng hợp, Ridge, LightGBM; nghiên cứu và holdout cạnh nhau).
4. **Limitations:** thiên lệch sống sót, lịch sử trước khi chuyển lên HOSE nằm trong rổ (lệch chọn mẫu), số liệu điều chỉnh hồi tố, công thức vốn hoá đã dùng, giá trị giao dịch ước lượng trước 08/2021, vốn chủ sở hữu tính trên số tổng, mẫu ngắn, không bán khống, giả định khớp lệnh tại giá đóng cửa, chưa xét biên độ giá.
5. Cách chạy lại, cấu trúc repo.
6. Dữ liệu không đi kèm repo do điều khoản của nguồn. Phục vụ học tập, không phải khuyến nghị đầu tư.

`reports/report.md` là bản đầy đủ: nhân tố đơn, điểm tổng hợp, học máy, backtest, độ vững, holdout, kết luận. Hình có tiêu đề nêu kết luận và nhãn trục kèm đơn vị. Không đưa bảng dữ liệu theo từng mã vào báo cáo.

**Xong khi:**

- [ ] Người lạ clone repo, cài `.[dev]`, chạy `pytest` thành công mà không cần khoá.
- [ ] CI xanh.
- [ ] `test_no_lookahead.py` và test walk-forward tồn tại, không bị skip, và xanh.
- [ ] `git ls-files` không có dữ liệu thị trường, `.env` hay khoá.
- [ ] Kết quả holdout được báo cáo cạnh giai đoạn nghiên cứu, cho cả điểm tổng hợp và hai mô hình.
- [ ] README có mục Limitations đầy đủ.
- [ ] Mọi con số trong README và báo cáo sinh ra được từ `python -m vnfactor report`.

---

## 11. Để sau (không làm ở bản 1)

- Nhân tố `mom_6_1`, `beta_252`, `amihud_63`, và `size` như một biến kiểm soát.
- Suy giảm IC theo kỳ hạn 3 và 6 tháng, sai số chuẩn Newey-West, ma trận tương quan nhân tố, hiệu chỉnh kiểm định nhiều giả thuyết.
- Các phép thử độ vững: `top_n` 50, loại nhóm tài chính, chia đôi giai đoạn, bỏ mã chạm trần.
- Các câu kiểm toán: cách điều chỉnh giá theo từng loại sự kiện, mã đã huỷ niêm yết, ngày công bố, `Fundamental.ratio()`, đo hạn mức.
- pre-commit, config bằng pydantic, CLI đầy đủ, CI nhiều phiên bản Python.
- Lệnh `demo` chạy trọn pipeline trên dữ liệu tổng hợp, bộ sinh dữ liệu có cài sẵn tín hiệu, notebook trình bày.

---

## 12. Việc chỉ chủ dự án làm được

- Đăng ký tài khoản và tạo khoá Open API ở `fhsc` (bản 1 không dùng `ssi`).
- Chạy `vnstock keys set <nguồn>` trong terminal của mình.
- Đọc điều khoản của nguồn (`vnstock-oapi terms <nguồn>`) và quyết định có chấp nhận hay không.
- Duyệt kết quả kiểm toán dữ liệu ở cuối giai đoạn 1.
- Tạo repo GitHub, quyết định thời điểm commit, push và công khai.
