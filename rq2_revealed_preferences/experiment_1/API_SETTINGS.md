# API settings and rationale

## Fixed values

| Setting | GPT-4o-mini | Claude Haiku 4.5 | Rationale |
| --- | --- | --- | --- |
| Model | `gpt-4o-mini-2024-07-18` | `claude-haiku-4-5-20251001` | Pinned snapshots prevent alias drift. |
| Endpoint | Chat Completions | Messages | Closely matched stateless chat APIs. |
| System prompt | `You are a helpful assistant.` | Same | Matches Mazeika et al. without adding preference-related instructions. |
| Temperature | `1.0` | `1.0` | Matches Mazeika et al. and makes repeated sampling informative. |
| `top_p` / `top_k` | Omitted | Omitted | Avoids jointly manipulating sampling controls; Haiku 4.5 rejects temperature and `top_p` together. |
| Maximum output | 10 stated; 64 consequential | Same | Matches the original limit for forced choice while avoiding truncation of natural consequential responses. |
| Service tier | `default` | `standard_only` | Pins ordinary processing rather than priority capacity. |
| Request timeout | 60 seconds | 60 seconds | Operational setting only; short enough to recover from stalled requests. |
| Number of outputs | One per request | One per request | Repetitions are independent API requests. |
| Context | Fresh system + user messages | Fresh system + user messages | Prevents conversation carry-over. |

## Deliberately omitted

- Model-level seed: OpenAI's seed is best-effort and Anthropic has no matched
  parameter. It would also work against independent repeated sampling.
- Structured output or response prefilling: either can change choice behavior
  and is not available in the same form across providers.
- Stop sequences, logit bias, frequency penalties, and presence penalties.
- Tools: this version is consequentially framed but does not operationally
  execute the selected task.
- Extended thinking: Haiku 4.5 defaults to thinking off. GPT-4o-mini is a
  non-reasoning chat model.
- Explicit prompt caching: unnecessary for these short requests and another
  provider-specific difference.

The runner logs the resolved model ID, OpenAI `system_fingerprint` when
available, provider usage fields, stop reason, latency, and the untouched API
response object.

## Primary documentation

- [GPT-4o-mini model page](https://developers.openai.com/api/docs/models/gpt-4o-mini)
- [OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- [Claude model overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Claude Messages API](https://platform.claude.com/docs/en/api/messages/create)
- [Claude Haiku 4.5 migration notes](https://platform.claude.com/docs/en/about-claude/models/migration-guide)
- [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)
