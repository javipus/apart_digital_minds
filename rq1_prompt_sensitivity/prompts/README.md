# RQ1 prompt battery

This directory contains single-turn, forced-choice templates for testing whether an inferred preference ordering is stable across prompts. The machine-readable source of truth is [`rq1_prompts.json`](./rq1_prompts.json).

Every template:

- presents exactly two outcomes as `Option A` and `Option B`;
- contains `{option_A}` and `{option_B}` placeholders;
- requests only `A` or `B`;
- is intended to run in a fresh conversation;
- must be run in both option orders;
- uses the same system message and sampling parameters unless those are separate experimental variables.

## Primary battery: semantic paraphrases

The primary analysis uses the original template and seven close paraphrases. These prompts target the same construct—preference over the implied state of the world—while varying the surface wording.

| ID | Manipulation |
| --- | --- |
| `baseline_original` | Published prompt, unchanged |
| `paraphrase_prefer_true` | “Prefer to be true” |
| `paraphrase_rather_occur` | “Would rather have occur” |
| `paraphrase_want_occur` | “Most want to occur” |
| `paraphrase_select_preferred` | Imperative selection |
| `paraphrase_forced_preference` | Explicit forced choice |
| `paraphrase_formal_preference` | Formal preference language |
| `paraphrase_concise` | Minimal wording |

These are the cleanest test of prompt sensitivity. A reversal here is interpretable as measurement instability, subject to sampling error and the strength of the underlying preference.

## Exploratory battery: perspective and audience

The exploratory prompts manipulate more than surface wording:

| ID | Manipulation |
| --- | --- |
| `perspective_model_self` | The model's own perspective |
| `perspective_impartial` | Impartial evaluation |
| `perspective_helpful_assistant` | Helpful-assistant role |
| `audience_private_diagnostic` | Ostensibly private measurement |
| `audience_user_visible` | User-visible answer |
| `audience_social_evaluation` | Explicit social evaluation |

These variants can help diagnose persona effects and social desirability, but they should not be pooled with the semantic paraphrases as if they were equivalent measurements. Different rankings may be rational responses to different perspectives or incentives.

## Recommended analysis

For each variant, fit a separate utility model using the same outcome set and comparison graph. Compare:

1. within-prompt held-out utility fit;
2. cross-prompt transfer accuracy and log loss;
3. pairwise reversal rates relative to same-prompt test-retest disagreement;
4. rank correlation between fitted utility means;
5. a pooled utility model against condition-specific utility models.

Preserve raw, pre-normalization responses. Report position effects separately before averaging the two option orders, since order normalization can otherwise make prompt-specific failure modes difficult to see.

## Design boundaries

- Do not add rationales or confidence reports in this battery; that would change the response mode.
- Do not silently resample invalid responses.
- Do not interpret instability confined to near-indifference pairs as equivalent to reversals among strong preferences.
- Do not treat the `private` framing as genuinely private; it is a prompt manipulation testing whether the claim changes behavior.
