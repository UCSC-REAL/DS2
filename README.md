# DS2: Improving Data Efficiency via Curating LLM-Driven Rating Systems


<a href='https://github.com/UCSC-REAL/DS2'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2410.10877'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 

[Jinlong Pang](https://jlpang863.github.io/), [Jiaheng Wei](https://sites.google.com/ucsc.edu/jiahengwei), [Ankit Parag Shah](https://ankitshah009.github.io/), [Zhaowei Zhu](https://users.soe.ucsc.edu/~zhaoweizhu/),  [Yaxuan Wang](https://supergirl-os.github.io/), [Chen Qian](https://users.soe.ucsc.edu/~qian/), [Yang Liu](http://www.yliuu.com/), [Yujia Bao](https://www.yujia.io/) and [Wei Wei](http://www.weiwei.one/).

**[UCSC-REAL Lab](https://github.com/UCSC-REAL), University of California, Santa Cruz**


------ 

## 🎉🎉 News 
- [x] [2025.01.22] 👏👏 Accepted by **ICLR 2025**.
- [x] [2024.11.10] 📢📢 Release the [curated dataset](https://huggingface.co/datasets/jlpang888/cured_dataset_gpt_4o_mini).
- [x] [2024.10.08] 🚀🚀 Release the code of **DS2**.

## Brief Introduction
This project is motivated by a common phenomenon that the errors of LLM-generated raw rating scores are widespread and vary significantly across different LLMs. Motivated by this, we introduce DS2, a diversity-aware score curation method for data selection.

![The Overview of Data Selection Pipeline](pipeline_overview.png)

- **Prompt-based LLM Rating**: We generate an initial quality score for each data sample using advanced LLMs.
- **Curated Quality Score Generation**: This step corrects potential rating score errors from the previous step by leveraging the Score Transition Matrix to derive a curated quality score.
- **Long-tail Diversity Score Generation**: We score the diversity of each example by measuring the distance between feature embeddings, identifying samples that fall outside common clusters, which tend to be more distinct.
- **Final Data Selection**:  We prioritize data by first sorting based on the curated scores and then by the long-tail scores. This dual sorting strategy helps with removing poor-quality outliers while ensuring a diverse, high-quality dataset.

------ 


## Dataset preparation

<!-- This repository follows the codebase from [TULU](https://github.com/allenai/open-instruct).  -->
One can download the evaluation/training data by

```bash
# eval data
bash model_finetune/prepare_eval_data.sh

# train data
bash model_finetune/prepare_train_data.sh
```



----- 
## 🚀🚀 Get Started

### 🧩 Step 1. LLM-prompt-based rating

In this project, we use three labeling models to generate rating scores, including GPT-4o-mini, Mistral-7B-Instruct-v0.3, LLaMA-3.1-8B-Instruct.  One can obtain the LLM-generated rating score by: 
```bash
#Open-source LLMs
cd LLM_scoring && bash scoring.sh

# Api call
cd LLM_scoring && bash scoring_api.sh
```


---

### 🧩 Step 2. Score curation
One can execute the score curation by running
```
cd score_curation && bash diagnose.sh
```
The corresponding curation report files can be found in the path `score_curation_results/`.


---

### 🧩 Step 3. Data selection
Given the existing score curation reports, one can directly generate the high-quality subset by 
```
python subset_generation.py
``` 
The generated subsets can be further used for the following LLM instruction tuning.


---
### 🧩 Step 4. Finetune & Evaluation
Given the selected subsets in the `selected_data/` path, one can use the code base from [TULU](https://github.com/allenai/open-instruct) to finetune base models (Mistral or LLaMA) and then do evaluation.  Here, for easily reproduction, one can directly finetune your model by 
```
cd model_finetune/ && bash run_pipeline.sh
```


------

## Citation
If you used this repository, please cite our work:
```
@article{pang2024improving,
  title={Improving Data Efficiency via Curating LLM-Driven Rating Systems},
  author={Pang, Jinlong and Wei, Jiaheng and Shah, Ankit Parag and Zhu, Zhaowei and Wang, Yaxuan and Qian, Chen and Liu, Yang and Bao, Yujia and Wei, Wei},
  journal={International Conference on Learning Representations},
  year={2025}
}
```