# re-pilot 项目规范

## 实验纪律

（全局规则见 ~/.claude/CLAUDE.md「模型实验：自包含目录 + 实验台账」，本节是项目落点。）

### 一次实验 = 一个自包含目录

每个实验在 `runs/<YYYY-MM-DD>_<名字>/` 下，复现只依赖这个目录：

```
runs/2026-11-23_v2-37fam/
├── config.yaml     # 超参 + 语料 md5 指纹 + git hash + teacher/student 型号 + seed（跑前写死）
├── scripts/        # 当次用的训练/评测脚本快照 + 基线 commit（有未提交改动留 diff）
├── data/           # 当次训练语料快照；放不下则留指纹（hash/行数/切分/来源）
├── metrics.json    # 域内 / OOD 双基准 + 预立判据 + 判定
├── notes.md        # 方案 → 结果 → 思考 → 下一步建议
└── adapters/       # 权重与落点清单（不进 git：keeper 附 tag Release，归档附 archive-adapters Release）
```

- 新实验**先建目录再开跑**，config 跑前写死（跑完补的 config 不可信）
- 脚本从 `runs/<名>/config.yaml` 读参数、往该目录写产物，不往共享路径写
- 基准文件（`data/eval.jsonl`、`data/ood_real.jsonl`）全实验共用不复制；run 的 `data/` 只放该次**训练**语料
- 失败实验目录**保持不动**，不删不改名；权重可丢了再训（config + data 在，可精确再生）

### 台账（硬规则）

**每次实验开始前和结束后，都必须更新仓根 `EXPERIMENTS.md`**（索引；事实源在 run 目录）：

- **leaderboard**：顶部"当前最优"随更好的结果刷新，**旧记录永不删除**（被取代的版本留在"全版本"表）
- **log**："尝试记录"按时间倒序，**每条至少四段：方案 → 结果 → 思考 → 下一步建议**；结果段必含域内 / OOD 双基准，失败实验同样四段齐全
- **开始前**：登记方案与预立判据（域内 / OOD 各一个阈值），状态 `🔄 进行中`——没有判据不开跑
- **结束后**：回填结果、思考、下一步建议；刷新最优则同步 leaderboard

### 资产

- 当前 keeper 固定在 `adapters/` + `data/{train,eval,ood_real}.jsonl`，被取代时整体迁入对应 run 目录归档
- 权重不进 git（GitHub 单次 push ~2GB 软上限）：keeper 附版本 tag 的 Release，归档附 `archive-adapters` Release
- 双基准同报：`data/eval.jsonl`（域内）+ `data/ood_real.jsonl`（OOD，保持 held-out 纯净，新增场景族时排除与它重叠的）
- 清理只针对确认无用的生成物（缓存、量化中间文件）；**实验资产默认不动**

## Git 纪律

- **禁止 `git add -A` / `git add .`**：只 add 点名的文件。本项目三次翻车：checkpoints 入库、fused_model 3.76GB 触发 GitHub 100MB 硬限、`mv` 改名 + `add -A` 把 keeper 权重从仓库删掉。新产物目录先补 `.gitignore`
- 后台长任务：日志直落文件 + 末尾显式 `EXIT=$?`，不要管道接 tail/grep（会吃掉退出码与进度）
- 发版按全局完备性检查：pyproject 版本 = tag、annotated tag、Release 四段 notes、tarball 实测 200

## 写 Python

- 工具链：`uv run ruff check . && uv run ruff format --check . && uv run mypy .`（strict），提交前三清
- 国内网络：`UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`；权重走 `HF_ENDPOINT=https://hf-mirror.com`，大文件用 curl 直灌后按 content-length 字节校验（hub 客户端对 302→xet CDN 会静默假成功）
- Qwen3/Qwen3.5 是 thinking 模型：所有生成调用传 `enable_thinking=False`，否则 `<think>` 段烧光 token 上限并毁掉 JSON
- `experiments/` 下的代码也要过 mypy strict，不因为是实验就豁免
