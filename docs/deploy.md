# 服务端部署调研与方案（GGUF 路线）

> 2026-11-22 调研定稿：把 2B 蒸馏模型部署为服务器上的共享服务（OpenAI 兼容 API）。
> 前置事实：MLX 是 Mac 专属栈，上 Linux 服务器必须跨栈转换。

## 格式选型：GGUF（llama.cpp）而非 ONNX

| 维度 | GGUF | ONNX (ORT GenAI) |
|---|---|---|
| 内存 | Q4_K_M 约 1.3GB（2B），k-quant 同尺寸质量最好 | int4 类似，同精度质量略逊 |
| CPU 推理速度 | llama.cpp 解码内核（AVX2/NEON）业界最快 | LLM 解码路径普遍更慢 |
| 成熟度 | LLM 服务最成熟的 CPU 栈：KV cache/采样/chat template/OpenAI API 全内置，单二进制 | ORT 本体稳，但生成式路径生态年轻、需自己拼 |
| 例外场景 | — | .NET/Java 宿主嵌入、WASM、非生成式模型（本项目不沾） |

## Qwen3.5 特有的三个注意点

1. llama.cpp 2026 年才支持 Qwen3.5，**必须最新版**；早期构建（b8234 附近）有吞吐量 bug，部署后要 benchmark
2. Qwen3.5 多模态原生，GGUF 带独立 mmproj 视觉文件——Ollama 目前跑不了，llama.cpp 纯文本 GGUF 没问题；转换剥离视觉可再省体积
3. 利好：llama.cpp 2026-05 合入 MTP 多 token 预测（1.5-2 倍生成提速）

## 部署链（v2/v3 定型后执行）

```
MLX fuse 导出 HF 格式（剥离视觉）
→ convert_hf_to_gguf.py 转纯文本 GGUF
→ 量化 Q4_K_M（约 1.3GB）
→ 等价性验证门禁：同一评测集重跑，91% 不掉才上线（掉了换 Q6 或回落 vLLM）
→ llama-server 起 OpenAI 兼容 API
→ 服务端 re 验证器兜底
```

## 服务设计：不确定性靠架构围住

```
用户 POST {"requirement": "...", "samples": [...]}
→ 模型生成 regex（不确定的一步：自然语言→形式规格，无确定性算法的地带）
→ 服务端 re.compile + 样例执行（确定的一步）
→ 返回 {"regex", "validated": true/false, "sample_results"}
```

错误从"静默的坑"变成"带标记的候选"，91%/15% 的能力边界对用户透明。

## 参考

- [llama.cpp Qwen3.5 支持提示](https://www.reddit.com/r/LocalLLaMA/comments/1red6sv/update_your_llamacpp_for_qwen_35)
- [unsloth: Qwen3.5 本地运行指南（mmproj/Ollama 问题）](https://unsloth.ai/docs/models/qwen3.5)
- [Qwen 官方 llama.cpp 文档](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [Qwen3.5-0.8B-GGUF 吞吐 bug 报告](https://huggingface.co/AaryanK/Qwen3.5-0.8B-GGUF)
