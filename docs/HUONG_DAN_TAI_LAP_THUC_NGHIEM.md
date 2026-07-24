# Hướng dẫn tái lập và kiểm chứng thực nghiệm PlantXAI-Stability

Tài liệu này dành cho thành viên nhận mã nguồn và **bộ dữ liệu đã được chuẩn
bị, kiểm tra và đóng băng** từ trưởng nhóm. Thành viên không phải tải lại dữ
liệu gốc, tiền xử lý ảnh, tạo manifest, chia tập, xử lý trùng lặp hay chọn mức
severity.

Mục tiêu của quy trình là tạo được bằng chứng độc lập ở một môi trường khác:

1. mã nguồn và protocol vượt qua kiểm thử;
2. hai mô hình có thể được train lại từ đúng tập train/validation đã đóng băng;
3. checkpoint mới được kiểm định trên validation mà không đọc official test;
4. kết quả official test đã công bố được xác minh bằng hash, số dòng và phân
   tích thống kê đã đăng ký;
5. bảng, hình và bản tóm tắt có thể được tạo lại từ kết quả đã đóng băng.

> **Ranh giới quản trị quan trọng**
>
> `DR-CHECKPOINT-001` và `DR-TEST-001` khóa đúng SHA-256 của hai checkpoint
> chính thức. Vì vậy checkpoint mới do thành viên train **không được** đưa vào
> official test hiện tại. Ngoài ra, `DR-RECOVERY-001` cấm chạy lại hai baseline
> và phần ResNet50/Grad-CAM đã hoàn tất. Không checkout mã cũ để né các cổng
> này. Muốn thực hiện một chiến dịch train-to-test độc lập, trưởng nhóm phải
> đăng ký một replication campaign và Decision Record mới trước khi mở tập
> test.

## 1. Hai tuyến tái lập

| Tuyến | Dữ liệu được đọc | Kết quả | Trạng thái |
|---|---|---|---|
| A — train lại | train và validation đã đóng băng | checkpoint mới, lịch sử train, validation audit | Được phép |
| B — kiểm chứng kết quả chính thức | artifact kết quả đã đóng băng; không đọc ảnh test | hash, thống kê, bảng và hình | Được phép |
| Official test với checkpoint mới | ảnh official test | kết quả test mới | **Chưa được phép** |

Tuyến A chứng minh khả năng tái huấn luyện và chất lượng validation ở môi
trường khác. Tuyến B chứng minh kết quả được báo cáo đúng với chiến dịch
official test đã đăng ký. Hai tuyến không được nối với nhau bằng cách đưa
checkpoint mới sang official test.

## 2. Những thứ trưởng nhóm phải cung cấp

Mã nguồn Git không chứa dữ liệu, checkpoint hoặc run output. Trước khi bắt đầu,
thành viên cần được cấp quyền đọc một `PlantXAI-replication-bundle` nằm trên
Google Drive, object storage hoặc ổ đĩa dùng chung.

Cấu trúc tối thiểu nên là:

```text
PlantXAI-replication-bundle/
├── SHA256SUMS.txt
├── environment/
│   ├── pip-freeze.txt
│   └── runtime.txt
├── data/
│   ├── plantxai-frozen-recovery-v1/
│   │   ├── dataset_manifest.csv
│   │   ├── dataset_manifest.parquet
│   │   ├── train_split.csv
│   │   ├── validation_split.csv
│   │   ├── test_split.csv
│   │   ├── split_summary.json
│   │   ├── split_leakage_report.json
│   │   ├── freeze_record.json
│   │   └── recovery_binding_report.json
│   └── plantxai-manifest-v2/
│       └── images/
├── official/
│   ├── checkpoints/
│   │   ├── resnet50_best.pt
│   │   ├── resnet50_checkpoint_evidence.json
│   │   ├── efficientnet_b0_best.pt
│   │   └── efficientnet_b0_checkpoint_evidence.json
│   ├── validation-audits/
│   ├── g2-readiness-v1/
│   │   └── g2_readiness_report.json
│   ├── baselines/
│   │   ├── resnet50-baseline/
│   │   └── efficientnet-b0-baseline/
│   ├── joint/
│   │   ├── resnet50-joint-merged-v1/
│   │   └── efficientnet-b0-joint-merged-v1/
│   ├── analysis-support-audit-v4/
│   └── statistical-analysis-v2/
└── member-runs/
```

Tên thư mục ngoài có thể khác, nhưng các file bên trong phải giữ nguyên tên và
byte. Không sửa CSV bằng Excel vì thao tác mở/lưu có thể đổi encoding, kiểu số
hoặc ký tự xuống dòng và làm sai hash.

Nếu bộ artifact chưa có `SHA256SUMS.txt`, trưởng nhóm phải tạo và ký/xác nhận
inventory trước khi phân phối. Không tự suy đoán file thiếu và không tạo lại
manifest bằng cách quét thư mục ảnh.

## 3. Danh tính bất biến cần đối chiếu

| Đối tượng | SHA-256 hoặc giá trị mong đợi |
|---|---|
| Commit chứa reporting chính thức | `35851dbc1dd234e536e6b0b3d8a9c3dd34410e45` |
| Protocol G0B dùng để train | `7eb0814be8ffc1a19f54e2bec2d2ca0c84d7f4d869d99e28b69e6c9e0e84523b` |
| Protocol G2 hiện tại | `ceaef8a293b877c61a81f046dbcca5ec9abdf7092c38e43fb1aa6225e76d8b02` |
| `dataset_manifest.csv` | `323b48e3564708d566e0e9f5c346a07ef728828b2879fc1975e21ca32e024894` |
| Historical final `freeze_record.json` | `aed2e96afd2749250d4151780bb4002d198eb96d7433d2bef5b03d4a6ac9212d` |
| ResNet50 checkpoint chính thức | `b508abd2851c5f576131db0e47447624cd78f1e3204c2931f7928c266f0c7bfc` |
| EfficientNet-B0 checkpoint chính thức | `05b592f1ff7f4f2b4a757ae2564a088e3742555e20110ee33d19e563ff2fe60b` |
| G2 readiness report | `e80184dd2e6d7c55df66eb266a94ec33addda8048731fdaa26a8195f8e82a7bc` |
| ResNet50 baseline report | `d7f86d988dc09ef119a0c0f1a3f2ddad223e1e17846f14d5bc680855c46b0877` |
| EfficientNet-B0 baseline report | `02e57a201ff37c4537ec57e45a4df48d3161c2b0f4982fd2c202d85514005252` |
| ResNet50 merge report | `32610c640f3f35455bcdd998a3f0bb1a09eac2ff34f40dcdb51c0d86e8ac7c1e` |
| EfficientNet-B0 merge report | `0cc81bef79c9bf273eff753eef74fa737d35d6b8abef7951acd3fa2e6d534401` |
| Analysis support report | `f370b3c7ace79cd5523242831593522671844f6459d2fa344f21f829613c13ac` |
| Official analysis report | `68a9b47fddb2f203aa35a78645849f4e15c11379dbba6dfc79c9a188557294de` |

`freeze_record.json` trong physical recovery bundle có thể có hash khác
historical hash. Trường hợp đó chỉ hợp lệ khi đi kèm
`recovery_binding_report.json` khớp `DR-RECOVERY-001`. Không thay physical hash
cho historical hash trong báo cáo khoa học.

## 4. Fork về GitHub cá nhân và clone

Repository gốc:

```text
https://github.com/phucpt10/XAI
```

Trên GitHub, chọn **Fork** để tạo `https://github.com/<GITHUB_USER>/XAI`. Sau
đó clone fork và khai báo repository gốc là `upstream`:

```bash
git clone https://github.com/<GITHUB_USER>/XAI.git
cd XAI
git remote add upstream https://github.com/phucpt10/XAI.git
git fetch --all --tags
git remote -v
```

Không push trực tiếp vào `main`. Tạo nhánh lưu ghi chú hoặc script cá nhân:

```bash
git switch -c reproduction/<TEN_THANH_VIEN>
git push -u origin reproduction/<TEN_THANH_VIEN>
```

Dataset, checkpoint và output phải nằm ngoài repository. `.gitignore` đã loại
trừ các đuôi checkpoint phổ biến, nhưng thành viên vẫn phải kiểm tra trước mỗi
lần commit:

```bash
git status --short
```

## 5. Chuẩn bị môi trường

Khuyến nghị dùng Linux/Google Colab với GPU CUDA cho train. Các bước phân tích
và reporting chỉ cần CPU. Dự án yêu cầu Python `>=3.10`.

Lần cài đặt đầu cần truy cập package index. Lần khởi tạo model còn cần
torchvision tải `IMAGENET1K_V2` cho ResNet50 và `IMAGENET1K_V1` cho
EfficientNet-B0. Nếu môi trường đích không có Internet, trưởng nhóm phải cung
cấp sẵn wheelhouse và cache pretrained weights trong bundle; không thay bằng
một bộ pretrained weights khác.

### 5.1. Khai báo đường dẫn

Ví dụ trong terminal Linux:

```bash
export PXAI_REPO="$PWD"
export PXAI_BUNDLE="/duong-dan/PlantXAI-replication-bundle"
export PXAI_FREEZE="$PXAI_BUNDLE/data/plantxai-frozen-recovery-v1"
export PXAI_IMAGES="$PXAI_BUNDLE/data/plantxai-manifest-v2"
export PXAI_OFFICIAL="$PXAI_BUNDLE/official"
export PXAI_RUNS="$PXAI_BUNDLE/member-runs/<TEN_THANH_VIEN>"
mkdir -p "$PXAI_RUNS"
```

Trong Colab, mount Drive trước, rồi thay `/duong-dan/...` bằng đường dẫn dưới
`/content/drive/MyDrive/...`. Đặt biến bằng `%env`, ví dụ:

```python
%env PXAI_REPO=/content/XAI
%env PXAI_BUNDLE=/content/drive/MyDrive/PlantXAI-replication-bundle
%env PXAI_FREEZE=/content/drive/MyDrive/PlantXAI-replication-bundle/data/plantxai-frozen-recovery-v1
%env PXAI_IMAGES=/content/drive/MyDrive/PlantXAI-replication-bundle/data/plantxai-manifest-v2
%env PXAI_OFFICIAL=/content/drive/MyDrive/PlantXAI-replication-bundle/official
%env PXAI_RUNS=/content/drive/MyDrive/PlantXAI-replication-bundle/member-runs/<TEN_THANH_VIEN>
```

Trong PowerShell dùng cú pháp `$env:PXAI_BUNDLE = "D:\..."`.

### 5.2. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hf,ml,xai,report,dev]"
```

Nếu cần byte-level reproducibility, cài đúng phiên bản trong
`environment/pip-freeze.txt` do trưởng nhóm cung cấp. Các cận phiên bản trong
`pyproject.toml` đủ để chạy chức năng nhưng không cam kết checkpoint giống từng
byte giữa các phiên bản CUDA/cuDNN/PyTorch khác nhau.

Lưu bằng chứng môi trường trước khi chạy:

```bash
python --version
python -m pip freeze > "$PXAI_RUNS/pip-freeze-before-run.txt"
nvidia-smi > "$PXAI_RUNS/nvidia-smi-before-run.txt"
git rev-parse HEAD > "$PXAI_RUNS/git-commit-before-run.txt"
```

Nếu không có GPU, bỏ lệnh `nvidia-smi`; không ghi một GPU giả vào báo cáo.

## 6. Gate 0 — kiểm tra code và protocol hiện tại

Checkout đúng commit reporting:

```bash
git switch --detach 35851dbc1dd234e536e6b0b3d8a9c3dd34410e45
python -m pip install -e ".[hf,ml,xai,report,dev]"
python -m plantxai_stability.cli validate-protocol \
  configs/protocol/v0.9/protocol.yaml
python -m plantxai_stability.cli smoke \
  configs/protocol/v0.9/protocol.yaml
python -m pytest
```

Điều kiện đạt:

- protocol hash là `ceaef8a...d8b02`;
- trạng thái là `frozen: true`;
- smoke test tạo đúng 12 scenario;
- toàn bộ test khả dụng đều PASS. Máy không cài PyTorch có thể skip integration
  test liên quan PyTorch, nhưng máy dùng cho train không được thiếu PyTorch.

## 7. Gate 1 — kiểm tra bundle đã chuẩn bị

Đầu tiên xác minh toàn bộ `SHA256SUMS.txt` bằng công cụ phù hợp với hệ điều
hành. Trên Linux:

```bash
cd "$PXAI_BUNDLE"
sha256sum --check SHA256SUMS.txt
cd "$PXAI_REPO"
```

Kiểm tra riêng manifest:

```bash
python -c "from plantxai_stability.provenance import sha256_file; \
p=r'$PXAI_FREEZE/dataset_manifest.csv'; \
print(sha256_file(p)); \
assert sha256_file(p)=='323b48e3564708d566e0e9f5c346a07ef728828b2879fc1975e21ca32e024894'"
```

Kiểm tra count/split mà không mở ảnh:

```bash
python -c "from collections import Counter; \
from plantxai_stability.data.manifest import read_manifest_csv; \
r=read_manifest_csv(r'$PXAI_FREEZE/dataset_manifest.csv'); \
c=Counter(x.split for x in r); print(len(r), c); \
assert len(r)==8384 and c=={'train':5328,'validation':1363,'test':1693}"
```

Nếu bất kỳ hash hoặc count nào sai, dừng tại đây. Không sửa file để làm cho
script chạy qua.

## 8. Tuyến A — train lại từ dữ liệu đã đóng băng

Training chính thức ban đầu chạy tại commit G0B
`3facdc158bc660b8138dc7d80447c99c85617841`. Checkout đúng commit này để
protocol có cùng training hash với freeze:

```bash
git switch --detach 3facdc158bc660b8138dc7d80447c99c85617841
python -m pip install -e ".[hf,ml,xai,dev]"
python -m plantxai_stability.cli validate-protocol \
  configs/protocol/v0.9/protocol.yaml
```

Protocol hash ở bước này phải là `7eb0814b...e84523b`.

### 8.1. Train ResNet50

```bash
python scripts/train_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest "$PXAI_FREEZE/dataset_manifest.csv" \
  --image-root "$PXAI_IMAGES" \
  --model-id resnet50 \
  --output-dir "$PXAI_RUNS/training/resnet50-v1" \
  --num-workers 0 \
  --device cuda
```

Nếu runtime bị ngắt, chạy lại đúng lệnh và thêm `--resume`. Không đổi commit,
manifest, protocol, model hoặc tham số:

```bash
python scripts/train_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest "$PXAI_FREEZE/dataset_manifest.csv" \
  --image-root "$PXAI_IMAGES" \
  --model-id resnet50 \
  --output-dir "$PXAI_RUNS/training/resnet50-v1" \
  --num-workers 0 \
  --device cuda \
  --resume
```

### 8.2. Train EfficientNet-B0

```bash
python scripts/train_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest "$PXAI_FREEZE/dataset_manifest.csv" \
  --image-root "$PXAI_IMAGES" \
  --model-id efficientnet_b0 \
  --output-dir "$PXAI_RUNS/training/efficientnet-b0-v1" \
  --num-workers 0 \
  --device cuda
```

Mỗi thư mục train phải có:

- `<model>_best.pt`;
- `<model>_latest.pt`;
- `<model>_history.json`;
- `<model>_checkpoint_evidence.json`.

Runner chỉ chọn checkpoint theo `validation_macro_f1`; tập test không được đọc.
Không dùng `--allow-draft-training` cho bằng chứng tái lập này.

### 8.3. Kiểm định checkpoint mới trên validation

Checkout commit đã bổ sung validation-only audit. Protocol không đổi so với
G0B:

```bash
git switch --detach eb674ac01250998e718ec8ee817c5468926b3d87
python -m pip install -e ".[hf,ml,xai,dev]"
```

ResNet50:

```bash
python scripts/audit_validation_checkpoint.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest "$PXAI_FREEZE/dataset_manifest.csv" \
  --image-root "$PXAI_IMAGES" \
  --model-id resnet50 \
  --checkpoint "$PXAI_RUNS/training/resnet50-v1/resnet50_best.pt" \
  --checkpoint-evidence "$PXAI_RUNS/training/resnet50-v1/resnet50_checkpoint_evidence.json" \
  --output-dir "$PXAI_RUNS/validation-audit/resnet50-v1" \
  --num-workers 0 \
  --device cuda
```

EfficientNet-B0:

```bash
python scripts/audit_validation_checkpoint.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest "$PXAI_FREEZE/dataset_manifest.csv" \
  --image-root "$PXAI_IMAGES" \
  --model-id efficientnet_b0 \
  --checkpoint "$PXAI_RUNS/training/efficientnet-b0-v1/efficientnet_b0_best.pt" \
  --checkpoint-evidence "$PXAI_RUNS/training/efficientnet-b0-v1/efficientnet_b0_checkpoint_evidence.json" \
  --output-dir "$PXAI_RUNS/validation-audit/efficientnet-b0-v1" \
  --num-workers 0 \
  --device cuda
```

Mỗi lệnh phải kết thúc bằng `Validation checkpoint audit: PASS`. Trong
`validation_checkpoint_audit.json`, kiểm tra:

```text
source_split: validation
test_split_accessed: false
sample_coverage_exact: true
selected_macro_f1_reproduced: true
```

Giá trị tham chiếu của chiến dịch gốc:

| Model | Best epoch | Validation macro-F1 | Validation accuracy |
|---|---:|---:|---:|
| ResNet50 | 29 | 0.9929163658 | 0.9941305943 |
| EfficientNet-B0 | 19 | 0.9971833207 | 0.9977989729 |

Checkpoint mới chỉ được coi là byte-identical nếu SHA-256 trùng checkpoint
chính thức. Nếu không trùng, báo cáo metric, epoch, runtime và sai khác một cách
trung thực. Mọi ngưỡng chấp nhận metric phải được trưởng nhóm đăng ký **trước**
khi xem kết quả run mới; không đặt tolerance sau khi đã thấy kết quả.

## 9. Tuyến B — kiểm chứng official test đã đóng băng

Quay lại commit reporting:

```bash
git switch --detach 35851dbc1dd234e536e6b0b3d8a9c3dd34410e45
python -m pip install -e ".[hf,ml,xai,report,dev]"
```

### 9.1. Xác minh baseline

Không chạy lại `scripts/evaluate_baseline.py`: `DR-RECOVERY-001` chủ động chặn
việc đó. Thay vào đó, kiểm tra hai `baseline_metrics.json` trong bundle bằng
`SHA256SUMS.txt` và các hash ở Mục 3.

Kết quả tham chiếu phải là:

| Model | Test samples | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|---:|
| ResNet50 | 1,693 | 0.995275 | 0.993991 | 8 |
| EfficientNet-B0 | 1,693 | 0.995865 | 0.995467 | 7 |

Các con số này là mô tả của hai checkpoint đã phê duyệt, không phải tiêu chí
để chọn lại checkpoint.

### 9.2. Xác minh hai joint merge

Kiểm tra hash của:

```text
official/joint/resnet50-joint-merged-v1/joint_merge_report.json
official/joint/efficientnet-b0-joint-merged-v1/joint_merge_report.json
```

Mỗi model phải có:

- `20,316` prediction rows = `1,693 × 12`;
- `60,948` joint rows = `1,693 × 12 × 3`;
- đủ 12 scenario và 3 CAM method;
- các exclusion được giữ lại, không xóa dòng thiếu metric.

Không chạy lại `run_joint_campaign_colab.py` trong một output rỗng. Chiến dịch
đã hoàn tất và recovery policy chỉ cho phép resume các phần chưa hoàn tất tại
thời điểm xảy ra sự cố, không cho tạo một official campaign thứ hai.

### 9.3. Tái tính phân tích thống kê để đối chiếu

Bước này không đọc ảnh và chỉ dùng hai merged result tree. Dùng commit đã tạo
official analysis:

```bash
git switch --detach 7eaeb4f884531de9e52ccc7856dfda95b81d176e
python -m pip install -e ".[hf,ml,xai,dev]"

python scripts/analyze_official_results.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --analysis-decision-record configs/protocol/v0.9/decision_records/DR-ANALYSIS-001.yaml \
  --analysis-support-decision-record configs/protocol/v0.9/decision_records/DR-ANALYSIS-SUPPORT-001.yaml \
  --analysis-support-audit-dir "$PXAI_OFFICIAL/analysis-support-audit-v4" \
  --resnet50-merge-dir "$PXAI_OFFICIAL/joint/resnet50-joint-merged-v1" \
  --efficientnet-b0-merge-dir "$PXAI_OFFICIAL/joint/efficientnet-b0-joint-merged-v1" \
  --output-dir "$PXAI_RUNS/statistical-analysis-reproduction-v1"
```

Lệnh phải kết thúc bằng `Official statistical analysis: PASS`. So sánh sáu CSV
với hash đã khóa trong `DR-RESULTS-001`:

| File | SHA-256 |
|---|---|
| `paired_comparisons.csv` | `1ef53d4eecb109872b379ae612e4d1cfd9540d7d20a835aac8ceb89b34ca7d9a` |
| `prediction_class_summary.csv` | `564f6505827eab074ac2903058d2f5abb33e696ca4742a4e774b252fb9d84f0a` |
| `prediction_summary.csv` | `65201d8003c1ad9d269e6c20522f81bb5e0c01f2f39fa2d7853f9058709121d6` |
| `rq3_association_summary.csv` | `f7e354bb6dac7e0ebb7f5d217e47a695ad1501282042ff1dc0fa8dab6c734a84` |
| `xai_exclusion_audit.csv` | `ecb9da6925f033e08e6cf9fec95410084798f93c0ae510f16b908e468c8c2ba9` |
| `xai_summary.csv` | `1a32f6f3b9f1d079a297ff1219c154222353a8bcae719b54af74b9aaf4bced74` |

`official_analysis_report.json` mới sẽ có timestamp và thông tin platform mới,
vì vậy không yêu cầu report hash mới trùng report gốc. Bằng chứng khoa học cần
đối chiếu là sáu child CSV, row count và acceptance criteria.

Row count mong đợi:

```text
prediction_summary_rows:       96
prediction_class_summary_rows: 480
xai_summary_rows:              432
paired_comparison_rows:        576
paired_estimable_rows:         573
paired_non_estimable_rows:       3
rq3_association_rows:           72
```

Ba dòng `Score-CAM × Gaussian blur severe` phải giữ trạng thái
`NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES`. Không diễn giải chúng là “không có
ý nghĩa thống kê”.

### 9.4. Tạo lại bảng, hình và tóm tắt công bố

Reporting phải đọc thư mục official `statistical-analysis-v2` được cung cấp,
không dùng thư mục analysis mới ở Mục 9.3 vì `DR-RESULTS-001` khóa đúng report
gốc.

```bash
git switch --detach 35851dbc1dd234e536e6b0b3d8a9c3dd34410e45
python -m pip install -e ".[report]"

python scripts/generate_frozen_results_report.py \
  --results-decision-record configs/protocol/v0.9/decision_records/DR-RESULTS-001.yaml \
  --analysis-dir "$PXAI_OFFICIAL/statistical-analysis-v2" \
  --output-dir "$PXAI_RUNS/frozen-results-reporting-v1"
```

Lệnh phải kết thúc bằng `Frozen results reporting: PASS` và tạo đúng:

- 8 bảng CSV;
- 6 hình PNG;
- `frozen_results_summary.json`;
- `frozen_results_summary.md`;
- `results_reporting_report.json`.

Reporting không mở ảnh, không chạy inference, không tính lại CAM và không thay
đổi kiểm định thống kê.

## 10. Các script không chạy trong quy trình “không tiền xử lý”

Không chạy các script sau:

```text
scripts/inspect_hf_dataset.py
scripts/audit_leaf_identity.py
scripts/adjudicate_quarantine.py
scripts/adjudicate_exact_duplicates.py
scripts/freeze_dataset.py
scripts/pilot_transform_severity.py
scripts/render_severity_review.py
```

Chúng thuộc giai đoạn tạo dữ liệu/protocol đã hoàn tất. Cũng không chạy lại:

```text
scripts/prepare_g2_readiness.py
scripts/evaluate_baseline.py
scripts/run_joint_campaign_colab.py
```

`prepare_g2_readiness.py` là cổng lịch sử trước khi G2 được phê duyệt; protocol
hiện tại đã ở G2. Hai runner official test còn lại bị ràng buộc bởi chiến dịch
đơn và recovery policy.

## 11. Hồ sơ bàn giao của thành viên

Sau khi hoàn tất, nộp một thư mục chỉ đọc gồm:

```text
<TEN_THANH_VIEN>-reproduction-evidence/
├── README.md
├── git-commit-before-run.txt
├── pip-freeze-before-run.txt
├── nvidia-smi-before-run.txt
├── bundle-sha256-check.txt
├── pytest.log
├── protocol-validation.log
├── training/
├── validation-audit/
├── statistical-analysis-reproduction-v1/
└── frozen-results-reporting-v1/
```

Trong `README.md` của hồ sơ ghi:

- ngày giờ và người thực hiện;
- hệ điều hành, Python, PyTorch, torchvision, CUDA/cuDNN và GPU;
- Git commit cho từng gate;
- đường dẫn/ID của bundle đầu vào và hash inventory;
- lệnh đã chạy, exit code và mọi lần resume;
- metric validation mới và sai khác so với tham chiếu;
- kết quả đối chiếu hash/row count;
- lỗi, warning hoặc sai khác chưa giải thích.

Không chỉ nộp ảnh chụp màn hình. Bằng chứng chính là log, JSON/CSV, hash và
thông tin lineage.

## 12. Tiêu chí kết luận

Một lần tái lập được ghi nhận là **PASS** khi:

1. bundle input vượt qua toàn bộ kiểm tra hash;
2. protocol và 12 scenario được xác minh;
3. test phần mềm PASS;
4. mỗi model train tạo đủ checkpoint/evidence và validation audit PASS;
5. validation audit xác nhận không truy cập test;
6. baseline và joint merge chính thức khớp hash/count đã đăng ký;
7. sáu CSV phân tích tái tính khớp hash hoặc mọi khác biệt được giải trình trước
   khi đưa ra kết luận;
8. reporting generator PASS và tạo đủ output allowlist;
9. không có tuning, reselection hoặc thay đổi protocol dựa trên official test.

Nếu mục tiêu là đánh giá official test cho checkpoint mới, kết luận phải là
`BLOCKED_PENDING_NEW_REPLICATION_CAMPAIGN`, không phải PASS và cũng không phải
lỗi kỹ thuật.
