# re-pilot 项目规范

## 实验纪律

- **仓库只保留 keeper**：`adapters/` + `data/{train,eval,ood_real}.jsonl` 是唯一受跟踪的模型资产。失败实验的权重与中间语料不入库、不留档——结论写进 `EXPERIMENTS.md` 一行即可
- **实验产物带版本后缀**（`adapters_v5/`、`data/train_v5.jsonl`），已被 gitignore 覆盖；确认失败即删，确认更优才替换 keeper
- **跑之前预立判据**：域内与 OOD 各一个阈值。没有判据不开跑——预立判据防的是失败的伪装
- **双基准同报**：`data/eval.jsonl`（域内）+ `data/ood_real.jsonl`（OOD）。OOD 基准保持 held-out 纯净，新增场景族时排除与它重叠的
- 刷新最优时同步更新 `EXPERIMENTS.md` 的"当前最优"段

## Git 纪律

- **禁止 `git add -A` / `git add .`**：只 add 点名的文件。本项目三次翻车：checkpoints 入库、fused_model 3.76GB 触发 GitHub 100MB 硬限、`mv` 改名 + `add -A` 把 keeper 权重从仓库删掉。新产物目录先补 `.gitignore`
- 后台长任务：日志直落文件 + 末尾显式 `EXIT=$?`，不要管道接 tail/grep（会吃掉退出码与进度）
- 发版按全局完备性检查：pyproject 版本 = tag、annotated tag、Release 四段 notes、tarball 实测 200

## 写 Python

- 工具链：`uv run ruff check . && uv run ruff format --check . && uv run mypy .`（strict），提交前三清
- 国内网络：`UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`；权重走 `HF_ENDPOINT=https://hf-mirror.com`，大文件用 curl 直灌后按 content-length 字节校验（hub 客户端对 302→xet CDN 会静默假成功）
- Qwen3/Qwen3.5 是 thinking 模型：所有生成调用传 `enable_thinking=False`，否则 `<think>` 段烧光 token 上限并毁掉 JSON
- `experiments/` 下的代码也要过 mypy strict，不因为是实验就豁免
