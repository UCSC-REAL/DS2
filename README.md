# DS2: Improving Data Efficiency via Curating LLM-Driven Rating Systems


<a href='https://github.com/UCSC-REAL/DS2'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2410.10877'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 

[Jinlong Pang](https://jlpang863.github.io/), [Jiaheng Wei](https://sites.google.com/ucsc.edu/jiahengwei), [Ankit Parag Shah](https://ankitshah009.github.io/), [Zhaowei Zhu](https://users.soe.ucsc.edu/~zhaoweizhu/),  [Yaxuan Wang](https://supergirl-os.github.io/), [Chen Qian](https://users.soe.ucsc.edu/~qian/), [Yang Liu](http://www.yliuu.com/), [Yujia Bao](https://www.yujia.io/) and [Wei Wei](http://www.weiwei.one/).

**[UCSC-REAL Lab](https://github.com/UCSC-REAL), University of California, Santa Cruz**


------ 

## 🎉🎉 News 
- [x] [2025.02.01] 👏👏 Accepted by **ICLR 2025**.
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


## Environment Setup
To run training, evaluation, or inference for finetuned models, you need to install the required packages by running the following command (after installing pytorch):
```bash
pip install -r requirements.txt
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
Th score curation codebase is from [Docta](https://github.com/Docta-ai/docta) in the `./score_curation` path. You can execute the score curation by running
```
cd score_curation && bash diagnose.sh
```
The corresponding curation report files could be found in the path `./score_curation/results`.


---

### 🧩 Step 3. Data selection
Given the existing score curation reports, you can directly use the following jupyter notebooks to do data selection including all baselines: `data_generation.ipynb`. The generated subsets can be further used for LLM instruction tuning. Other selected datasets used for ablation study can be also generated from the following jupyter notebooks located in the `./score_curation` path: `data_gen_score_curation.ipynb` and `data_gen_data_scale.ipynb`. In particular, we use `data_gen_score_curation.ipynb` to generate subsets after curating machine-generated raw scores.



---
### 🧩 Step 4. Finetune & Evaluation
Given the selected subsets in the path `model_finetune/selected_data/`, you can use the code base from [TULU](https://github.com/allenai/open-instruct) to finetune base models (Mistral or LLaMA) and then do evaluation.
In particular, you can submit the jobs via launcher under the path `model_finetune/`. For example, you can submit the job by running the code 
```
cd model_finetune/ && launcher run job_pipeline_all.yaml
```


Futhermore, we can also execute the code locally, e.g.,  
```
cd model_finetune/ && bash run_pipeline_all.sh
```

One can present the final result by running 
```
python model_finetune/read_results.py
```

------

## Final results 
The final results of LLM judging compared with human-annotated dataset LIMA can be found in `lima_compare_plot.ipynb`. Moreover, for the tabular results, you can check the `reading_results.ipynb` jupyter notebook.

------

## Citation
If you used this repository, please cite our work:
```
@article{pang2024improving,
  title={Improving Data Efficiency via Curating LLM-Driven Rating Systems},
  author={Pang, Jinlong and Wei, Jiaheng and Shah, Ankit Parag and Zhu, Zhaowei and Wang, Yaxuan and Qian, Chen and Liu, Yang and Bao, Yujia and Wei, Wei},
  journal={arXiv preprint arXiv:2410.10877},
  year={2024}
}
```