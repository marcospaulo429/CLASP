# CLASP Evaluation Roadmap

Planned testing and research directions. (Relocated from `src/clasp/data/testes.md`.)

## Noise robustness evaluation

- Choose which dataset to evaluate on (Spoken SQuAD, Speech Brown).
- Run the evaluation script on the chosen dataset to assess model performance under noisy conditions.
- Compare **ambient noise** vs. **white noise** vs. **reverberation**.
- Establish an **oracle topline** as a benchmark upper bound.
- Metrics to track: Mel Distance, PESQ, WER, EER, AAC.

## Codec analysis under high noise

- Select a set of codecs to evaluate (e.g. Opus, Speex, G.711).
- Encode and decode the audio samples with each codec under high-noise conditions.
- Analyze the impact of each codec on audio quality and model performance using
  Mel Distance, PESQ, WER, EER, and AAC.

## Best configuration under noise

- Compare the noise-robustness and codec-analysis results to determine which
  configurations perform best under noisy conditions.
- Identify trade-offs between audio quality and model performance across configurations.

## Codec impact on retrieval

- Evaluate retrieval performance on audio processed with different codecs.
- Analyze how codec choice affects retrieval, particularly in noisy environments.
- Use WER and EER to assess the impact of codecs on retrieval accuracy.

## MSEB / SVQ tasks

- Evaluate CLASP on the [Massive Sound Embedding Benchmark (MSEB)](https://github.com/google-research/mseb)
  using the [Simple Voice Questions (SVQ)](https://huggingface.co/datasets/google/svq) dataset.
- **Acoustic Hypothesis Reranking** — reorder phonetically-confusable text candidates by
  relevance to the spoken query (MAP / MRR / WER / CER).
- **SVQ audio→text retrieval** — verify CLASP retrieval on the new dataset.
- See [MSEB.md](MSEB.md) for how to run these.
