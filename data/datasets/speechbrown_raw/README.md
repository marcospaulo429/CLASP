---
license: mit  
language:  
- en  
pretty_name: Speech Brown  
size_categories:  
- 10K<n<100K  
task_categories:  
- text-to-speech  

---
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.13071) [![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/language-modeling-lab/CLASP) [![Website](https://img.shields.io/website?url=https%3A%2F%2Fmultimodalrag.github.io%2F)](https://clasp1.github.io/)


[Models](https://huggingface.co/llm-lab/CLASP) | [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-88717-8_2) | [arXiv Link](https://arxiv.org/abs/2412.13071) | [Proposed Dataset](https://huggingface.co/datasets/llm-lab/SpeechBrown)  | [ACM Digital Library](https://dl.acm.org/doi/10.1007/978-3-031-88717-8_2) | [Website](https://clasp1.github.io/)

## Dataset Summary

**Speech Brown** is a comprehensive, synthetic, and diverse paired speech-text dataset in 15 categories, covering a wide range of topics from fiction to religion. This dataset consists of over 55,000 sentence-level samples.  

To train the [CLASP](https://huggingface.co/llm-lab/CLASP) model, we created this dataset based on the Brown Corpus. The synthetic speech was generated using the [NVIDIA Tacotron 2](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/) text-to-speech model.  

For more information about our proposed model, please refer to this [paper](https://arxiv.org/abs/2412.13071) which is published at **ECIR 2025**. The dataset generation pipeline, along with code and usage instructions, is available on this [GitHub page](https://github.com/language-modeling-lab/CLASP).  


![image/png](https://cdn-uploads.huggingface.co/production/uploads/64ba58d377dd483716aba098/BT_bmv19WNz8OIXFcIWg5.png)
## Dataset Statistics
1. Total size: Approximately 30 GB.  
2. Number of samples: 55,173 pairs of speech and text.  
3. Average tokens per sample: 19.00.  
4. Maximum tokens in a sample: 48.  
5. Average characters per sample: 96.72.
6. Number of unique tokens: 50,667
7. Categories: 15 categories consist of `adventure`, `belles_lettres`, `editorial`, `fiction`, `government`, `hobbies`, `humor`, `learned`, `lore`, `mystery`, `news`, `religion`, `reviews`, `romance`, `science_fiction`.  

## Dataset Structure
To ensure ease of use, the dataset is partitioned into 10 parts. Each part can be used independently if it meets the requirements of your task and model.  

### Metadata Files
1. **global_metadata**: A JSON file containing metadata for all 55,173 samples.  
2. **localized_metadata**: A JSON file containing metadata for all samples, categorized into the 10 dataset partitions.  

### Metadata Fields
1. **id**: The unique identifier for the sample.  
2. **audio_file_path**: The file path for the audio in the dataset.  
3. **category**: The category of the sample's text.  
4. **text**: The corresponding text of the audio file.

## Usage Instructions

To use this dataset, download the parts and metadata files as follows:

#### Option 1: Manual Download
Visit the [dataset repository](https://huggingface.co/datasets/llm-lab/SpeechBrown/tree/main) and download all `dataset_partX.zip` files and the `global_metadata.json` file.

#### Option 2: Programmatic Download
Use the `huggingface_hub` library to download the files programmatically:

```python
from huggingface_hub import hf_hub_download
from zipfile import ZipFile
import os
import json

# Download dataset parts
zip_file_path1 = hf_hub_download(repo_id="llm-lab/SpeechBrown", filename="dataset_part1.zip", repo_type="dataset")
zip_file_path2 = hf_hub_download(repo_id="llm-lab/SpeechBrown", filename="dataset_part2.zip", repo_type="dataset")
# Download other parts...

# Download metadata
metadata_file_path = hf_hub_download(repo_id="llm-lab/SpeechBrown", filename="global_metadata.json", repo_type="dataset")

for i in range(1, 11):
    with ZipFile(f'dataset_part{i}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'dataset_part{i}')
    os.remove(f'dataset_part{i}.zip')

with open('global_metadata.json', 'r') as f:
    metadata = json.load(f)
metadata.keys()
```

## Citations
If you find our paper, code, data, or models useful, please cite the paper:  
```
@inproceedings{10.1007/978-3-031-88717-8_2,
                author = {Abootorabi, Mohammad Mahdi and Asgari, Ehsaneddin},
                title = {CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval},
                year = {2025},
                isbn = {978-3-031-88716-1},
                publisher = {Springer-Verlag},
                address = {Berlin, Heidelberg},
                url = {https://doi.org/10.1007/978-3-031-88717-8_2},
                doi = {10.1007/978-3-031-88717-8_2},
                abstract = {This study introduces CLASP (Contrastive Language-Speech Pretraining), a multilingual, multimodal representation tailored for audio-text information retrieval. CLASP leverages the synergy between spoken content and textual data. During training, we utilize our newly introduced speech-text dataset, which encompasses 15 diverse categories ranging from fiction to religion. CLASP’s audio component integrates audio spectrograms with a pre-trained self-supervised speech model, while its language encoding counterpart employs a sentence encoder pre-trained on over 100 languages. This unified lightweight model bridges the gap between various modalities and languages, enhancing its effectiveness in handling and retrieving multilingual and multimodal data. Our evaluations across multiple languages demonstrate that CLASP establishes new benchmarks in HITS@1, MRR, and meanR metrics, outperforming traditional ASR-based retrieval methods that rely on transcribing speech into text for subsequent text retrieval, especially in specific scenarios.},
                booktitle = {Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part IV},
                pages = {10–20},
                numpages = {11},
                keywords = {Multimodal IR, Speech Retrieval, Contrastive Learning},
                location = {Lucca, Italy}
}
```

## Contact
If you have questions, please email mahdi.abootorabi2@gmail.com or asgari@berkeley.edu.