steps in new testing: 
  - Noise Robustness Evaluation:
    - Choose which dataset to use for evaluation (Spoken Squad, Speech Brown).
    - Run the evaluation script on the chosen dataset to assess the model's performance under noisy conditions.
    - Ambient Noise x White Noise x Reverberation
    - Oracle topline for benchmark
    - Data to check: Mel Distance, PESQ, WER, EER, AAC

  - Codecs analisis in high noise:
    - Select a set of codecs to evaluate (e.g., Opus, Speex, G.711).
    - Encode and decode the audio samples using each codec under high noise conditions.
    - Analyze the impact of each codec on the audio quality and model performance using metrics such as Mel Distance, PESQ, WER, EER, AAC.
  
   - Comparison of which configuration works best with noise:
    - Compare the results from the noise robustness evaluation and codecs analysis to determine which configurations yield the best performance under noisy conditions.
    - Identify any trade-offs between audio quality and model performance for different configurations.

  - Study the improvement of codecs in retrieval tasks:
    - Evaluate the performance of the model on retrieval tasks using audio processed with different codecs.
    - Analyze how the choice of codec affects retrieval performance, particularly in noisy environments.
    - Use metrics such as WER and EER to assess the impact of codecs on retrieval accuracy.