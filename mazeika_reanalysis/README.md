# Mazeika preference-coherence reanalysis

This folder contains a first-pass replacement of the MMLU x-axis in Mazeika
et al.'s *Coherent Preferences Emerge With Scale* analysis with the
[Epoch Capabilities Index (ECI)](https://epoch.ai/eci).

The preference-coherence outcome is the paper's precomputed held-out accuracy
for its Thurstonian utility model. It is read from the summary files in the
published `options_hierarchical.zip` archive. The capability measure is read
from Epoch AI's precomputed `epoch_capabilities_index.csv` table.

Source snapshots were retrieved on 2026-08-15 from the
[Mazeika et al. data archive](https://huggingface.co/mmazeika/emergent-values-data/resolve/main/options_hierarchical.zip),
the paper's [source repository](https://github.com/centerforaisafety/emergent-values),
and Epoch's [Benchmarking Hub archive](https://epoch.ai/data/benchmark_data.zip).
Epoch's included README describes its data as CC BY 4.0.

Run from this directory:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
  python3 analysis/plot_eci_preference_coherence.py
```

Outputs:

- `results/eci_preference_coherence.png` and `.pdf`: annotated scatter plot and
  OLS fit.
- `results/eci_preference_coherence.csv`: the 12-model plotted intersection.
- `results/eci_model_coverage.csv`: all 32 paper models, including exclusions.
- `results/eci_preference_coherence_summary.json`: statistics and provenance.

The model join is explicit in the script. Eight rows are direct version
matches. The plot also includes four transparent first-pass matches with
distinct hollow markers: two inferred OpenAI API snapshots, one unpinned alias
with a single scored ECI snapshot, and one FP8/non-FP8 precision variant. The
summary JSON reports an exact-match-only sensitivity correlation, the original
paper-sample MMLU correlation, and MMLU on the same 12-model ECI overlap.
