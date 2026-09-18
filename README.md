# tinyserve-llm

A mini vLLM built around its KV cache — a small, readable inference engine in which memory
management is the first-class concern.

> **Status:** in development. Block allocator, copy-on-write and the KV write path are
> implemented. Prefix cache, offload, scheduler and engine are in progress, and the benchmark
> numbers below are placeholders until measured.

## Key Features

* 🔍 **Token-granular prefix reuse** — reuses a cached prefix down to the token, not just whole blocks
* 🌿 **Copy-on-write forking** — parallel sampling and beam search share KV until they diverge
* 💾 **CPU offload** — moves KV to pinned host memory under pressure instead of recomputing it
* 🎯 **Cache-aware scheduling** — orders requests by cache hit ratio, with aging to bound starvation
* 🧩 **Clean boundary** — the engine talks to the cache through one explicit interface
* 📖 **Readable** — ~2,300 lines, with flash-attn doing the attention math

## Installation

```bash
pip install -e ".[cuda]" --no-build-isolation   # Linux, CUDA, Ampere or newer
pip install -e ".[dev]"                         # CPU only — allocator, prefix cache, tests
```

## Manual Download

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/models/Qwen3-0.6B
```

## Quick Start

```python
from tinyserve_llm import LLM, SamplingParams

llm = LLM("~/models/Qwen3-0.6B")
params = SamplingParams(temperature=0.8, max_tokens=256)

outputs = llm.generate(["Explain paged attention in one paragraph."], params)
print(outputs[0]["text"])
```

## Benchmark

**Configuration:** RTX 4090, Qwen3-0.6B, 256 sequences, input and output lengths sampled
uniformly from 100–1024 tokens.

| Engine | Output tokens | Time (s) | Throughput (tok/s) |
| ------------ | ------------- | -------- | ------------------ |
| vLLM | TBD | TBD | TBD |
| nano-vllm | TBD | TBD | TBD |
| tinyserve-llm | TBD | TBD | TBD |

Throughput is not what tinyserve-llm optimises for. The memory benchmarks are:

| Workload | Metric | nano-vllm | tinyserve-llm |
| ------------------------- | ------------------------- | --------- | ----------- |
| Multi-turn chat | prefill tokens recomputed | TBD | TBD |
| Half the KV memory budget | throughput retained | TBD | TBD |
| Parallel sampling, n=8 | blocks allocated | TBD | TBD |

## License

MIT. Model and sampling layers adapted from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
