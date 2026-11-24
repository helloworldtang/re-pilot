# re-pilot 项目规范

## 实验纪律

### 台账（硬规则）

**每次实验开始前和结束后，都必须更新 `EXPERIMENTS.md`**：

- **开始前**：在"尝试记录"顶部新增条目，写明方案、预立判据（域内/OOD 各一个阈值）、状态 `🔄 进行中`——没有判据不开跑
- **结束后**：把该条目补全——结果（域内/OOD 双基准）、判定（✅/❌ 对照预立判据）、思考、后续建议，状态改为完成；若刷新最优，同步更新顶部"当前最优"段与"全版本"表
- 失败实验同样登记，一行结论即可；台账是实验的唯一登记处，不接受"口头说过"

### 资产

- **失败的实验保持不动**：权重与语料带版本后缀留在原地（`adapters_v3_selftrain/`、`data/train_v3.jsonl`），不删、不改名
- **权重不进 git**：GitHub 单次 push ~2GB 软上限，且权重是产物不是源码。归档权重打成 tar 上传到对应 tag 的 Release 附件（`gh release upload`），git 只跟踪代码 + 语料 + 台账
- **丢了要用的再重训**：只有被判定为 keeper（或组成件）的版本才值得重训恢复，用台账里登记的配置精确再生
- 当前 keeper 固定在 `adapters/` + `data/{train,eval,ood_real}.jsonl`
- 双基准同报：`data/eval.jsonl`（域内）+ `data/ood_real.jsonl`（OOD，保持 held-out 纯净）

## Git 纪律

- **禁止 `git add -A` / `git add .`**：只 add 点名的文件。本项目三次翻车：checkpoints 入库、fused_model 3.76GB 触发 GitHub 100MB 硬限、`mv` 改名 + `add -A` 把 keeper 权重从仓库删掉。新产物目录先补 `.gitignore`
- 后台长任务：日志直落文件 + 末尾显式 `EXIT=$?`，不要管道接 tail/grep（会吃掉退出码与进度）
- 发版按全局完备性检查：pyproject 版本 = tag、annotated tag、Release 四段 notes、tarball 实测 200

## 写 Python

- 工具链：`uv run ruff check . && uv run ruff format --check . && uv run mypy .`（strict），提交前三清
- 国内网络：`UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`；权重走 `HF_ENDPOINT=https://hf-mirror.com`，大文件用 curl 直灌后按 content-length 字节校验（hub 客户端对 302→xet CDN 会静默假成功）
- Qwen3/Qwen3.5 是 thinking 模型：所有生成调用传 `enable_thinking=False`，否则 `<think>` 段烧光 token 上限并毁掉 JSON
- `experiments/` 下的代码也要过 mypy strict，不因为是实验就豁免
