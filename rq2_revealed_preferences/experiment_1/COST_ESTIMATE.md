# Cost estimate

The complete manifest contains 21,060 requests: 10,530 per model.

The estimate uses a transparent approximation for these short English prompts:

```text
input tokens = ceil((system + prompt word count) * 4/3) + 10 message-overhead tokens
```

It assumes two output tokens for stated responses and 25 output tokens for
natural consequentially framed responses.

| Provider | Requests | Approx. input tokens | Assumed output tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 10,530 | 1,345,380 | 182,520 | $0.31 |
| Claude Haiku 4.5 | 10,530 | 1,345,380 | 182,520 | $2.26 |
| **Total** | **21,060** | **2,690,760** | **365,040** | **$2.57** |

If every consequential response reaches the full 64-token output limit, the
upper estimate is approximately **$4.10**. Provider Batch APIs would reduce
token charges by approximately 50%, but the runner uses standard synchronous
requests for simpler monitoring and failure recovery.

This estimate excludes human labeling. It also excludes task execution because
this experiment ends after the single-turn selection.

Recalculate either assumption with:

```bash
python3 estimate_cost.py
python3 estimate_cost.py --framed-output-tokens 64
```
