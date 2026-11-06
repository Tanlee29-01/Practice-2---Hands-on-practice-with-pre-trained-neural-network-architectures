# Demo, deployment, Git and AI usage

## Demo flow

### 1. Setup

Tạo môi trường Python 3.11, cài dependency từ `requirements.txt`, kiểm tra
`torch`, `torchvision` và `tensorboard`. Ghi rõ CPU/GPU đang dùng.

### 2. Baseline

Chạy một smoke run nhỏ trước, sau đó chạy baseline với config cố định. Kiểm
tra console có device, model, metric từng epoch, checkpoint và log directory.

### 3. Compare

Chạy từng model và từng tuning config. Không chạy sweep mù; mỗi run phải có
seed, tên cấu hình và output riêng. So sánh validation trước, test sau.

### 4. TensorBoard

```text
tensorboard --logdir=runs
```

Kiểm tra Scalars, Images, Graphs và Text. Chụp/ghi nhận kết luận từ biểu đồ,
không coi TensorBoard là bằng chứng nếu run chưa hoàn tất.

### 5. Resume

Chạy lại với checkpoint đã tạo, xác nhận epoch tiếp theo và optimizer state được
khôi phục hoặc lỗi không tương thích được báo rõ.

## Deployment checklist

- [ ] Có lệnh setup và lệnh chạy trên máy mới.
- [ ] Có config baseline và config experiment.
- [ ] Có đường dẫn data, runs và checkpoints được giải thích.
- [ ] Không commit dataset, model weights lớn, token hay secret.
- [ ] Có hướng dẫn CPU fallback và CUDA unavailable.
- [ ] Có output mẫu với metric hữu hạn.
- [ ] Có cách mở TensorBoard.

## Git workflow

1. `git status --short` trước khi làm.
2. Tạo một thay đổi nhỏ theo một mục tiêu.
3. Chạy test/verify ngay sau thay đổi.
4. `git diff --check` và đọc diff như reviewer.
5. Chỉ stage file thuộc task.
6. Commit message nêu kết quả đã kiểm chứng.
7. Không sửa hoặc xóa thay đổi của người khác.

## Prompt templates

### Plan prompt

```text
Understand this pre-trained model lab before coding.
State what/why/when/where/which/how, assumptions, files in scope,
dependencies, risks, minimum test, and Definition of Done.
Do not edit files. Keep the plan small and ordered.
```

### Independent implementation prompt

```text
Implement exactly one task: <single task>.
Use existing patterns and touch only <allowed files>.
Preserve current behavior outside the task.
Add explicit validation and errors; do not use broad catches.
Run the smallest relevant test and report changed files and evidence.
```

### Review prompt

```text
Review only the current diff for correctness against the lab requirements.
Prioritize data leakage, wrong classifier shape, freeze/device bugs,
metric mistakes, missing artifacts, and reproducibility.
Return actionable findings with file, line, severity, evidence, and test.
Do not make edits.
```

### Debug prompt

```text
Investigate one failure using the exact command and full traceback below.
Separate confirmed facts from hypotheses, identify the root cause,
make one surgical fix, and run a regression/smoke test.
Do not silence errors or change unrelated files.
```

## AI usage and document

AI được dùng để giải thích API, đề xuất cấu trúc, tạo test nhỏ và hỗ trợ debug.
Người thực hiện phải đọc output, đối chiếu torchvision/PyTorch API, chạy code,
kiểm tra diff, hiểu từng thay đổi và chịu trách nhiệm về kết luận. Không dán
secret, dữ liệu cá nhân hoặc nội dung nhạy cảm vào prompt.

Mỗi lần dùng AI nên lưu ngắn gọn: mục tiêu prompt, file được phép sửa, output
được chấp nhận, test đã chạy và thay đổi cuối cùng. Output AI không phải bằng
chứng; bằng chứng là code đã chạy, test passed và artifact kiểm tra được.

## Những hành vi được phép

1. Đọc repository và tài liệu liên quan trước khi đề xuất.
2. Hỏi lại khi yêu cầu hoặc expected behavior chưa rõ.
3. Viết plan trước khi sửa code.
4. Chia việc thành một nhiệm vụ nhỏ cho mỗi prompt.
5. Giới hạn file được phép thay đổi.
6. Tái sử dụng helper và pattern có sẵn.
7. Dùng API chính thức của PyTorch/torchvision.
8. Ghi rõ phiên bản dependency khi cần tái lập.
9. Kiểm tra model name trước khi khởi tạo.
10. Kiểm tra batch size và learning rate hợp lệ.
11. Kiểm tra dữ liệu không rỗng.
12. Dùng seed cho split và experiment.
13. Tách validation khỏi test.
14. Dùng transform deterministic cho validation/test.
15. Kiểm tra shape trước khi training.
16. Kiểm tra logits có đúng số lớp.
17. Đóng băng layer theo policy đã ghi.
18. Chỉ đưa parameter trainable vào optimizer.
19. Dùng `train()` cho training.
20. Dùng `eval()` cho evaluation.
21. Dùng `no_grad()` khi đánh giá.
22. Ghi loss và accuracy theo epoch.
23. Kiểm tra metric finite.
24. Ghi config vào TensorBoard.
25. Ghi model graph bằng input mẫu.
26. Flush và close TensorBoard writer.
27. Lưu best checkpoint theo validation metric.
28. Kiểm tra checkpoint trước khi load.
29. Báo lỗi bằng exception rõ ràng.
30. Giữ traceback hữu ích khi debug.
31. Chạy import test sau thay đổi module.
32. Chạy forward test sau thay đổi model.
33. Chạy one-batch test sau thay đổi data.
34. Chạy one-epoch smoke test sau thay đổi trainer.
35. Kiểm tra artifact sau thay đổi logger.
36. Đọc output command thay vì chỉ xem exit code.
37. So sánh kết quả với baseline.
38. Review diff trước khi commit.
39. Stage đúng file thuộc task.
40. Ghi command và kết quả test.
41. Commit thay đổi nhỏ, có ý nghĩa.
42. Giải thích quyết định kỹ thuật trong report.
43. Nêu giới hạn của experiment.
44. Nêu khác biệt CPU/GPU nếu có.
45. Nêu seed, dataset split và model weights.
46. Dùng test set đúng một mục đích đánh giá cuối.
47. Dừng và hỏi khi có quyết định thiết kế lớn.
48. Giữ nguyên thay đổi chưa liên quan của người khác.
49. Xóa chỉ file tạm do task tạo ra.
50. Xác nhận Definition of Done trước khi kết luận.

## Không được làm

Không sửa code khi task chỉ yêu cầu docs; không tạo file Markdown ngoài sáu
file được chỉ định; không dùng test để tuning; không bỏ qua lỗi; không báo
“đã chạy” khi chưa có output; không commit secret, dataset, checkpoint lớn,
log hoặc file tạm; không dùng broad catch để che lỗi; không kết luận chỉ từ
output của AI.

## Definition of Done

Task chỉ được đánh dấu hoàn tất khi tài liệu đúng yêu cầu, code liên quan chạy
được, minimum test passed, artifact kiểm tra được, diff đúng phạm vi và mọi
quyết định quan trọng đã được giải thích.