# re-pilot 项目规范

## 实验纪律

- **每次训练尝试必须登记 `EXPERIMENTS.md`**：在"尝试记录"最上方插一条（方案 / 预立判据 / 结果 / 思考 / 后续建议）；若刷新最优，同步更新顶部"当前最优"段与全版本总表。失败也登记——负结果和成功同等重要
- **跑实验前先预立判据**（域内与 OOD 各一个阈值），写进台账条目；没有判据的实验不开跑——预立判据防的是失败的伪装（"域内还涨了 2pp"这类半成功叙事）
- **双基准**：域内 = `data/eval.jsonl`，OOD = `data/ood_real.jsonl`，两组同时报。OOD 基准必须保持 held-out 纯净——新增场景族时排除与它重叠的
- **适配器显式命名**：多版本并存时脚本点名具体版本（`adapters_v2_keeper`），不用"当前 adapters"这类相对指称；链脚本里的 `mv` 操作前先 `ls -d adapters*` 确认

## Git 纪律

- **禁止裸 `git add -A`**：先 `git status` 过目，确认无大文件（本项目已两次翻车：checkpoints、fused_model 3.76GB 触发 GitHub 100MB 硬限）。新模型产物目录（`models/`、`fused_model/`、`adapters*/`、`data/*.jsonl` 生成物）先补 `.gitignore`
- 后台长任务：日志直落文件 + 末尾显式 `EXIT=$?`，不要管道接 tail/grep（会吃掉退出码与进度）
- 发版按全局完备性检查：pyproject 版本 = tag、annotated tag、Release 四段 notes、tarball 实测 200

## 写 Python

- 工具链：`uv run ruff check . && uv run ruff format --check . && uv run mypy .`（strict），提交前三清
- 国内网络：`UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`；权重走 `HF_ENDPOINT=https://hf-mirror.com`，大文件用 curl 直灌后按 content-length 字节校验（hub 客户端对 302→xet CDN 会静默假成功）
- Qwen3/Qwen3.5 是 thinking 模型：所有生成调用传 `enable_thinking=False`，否则 `<think>` 段烧光 token 上限并毁掉 JSON
- `experiments/` 下的代码也要过 mypy strict，不因为是实验就豁免
