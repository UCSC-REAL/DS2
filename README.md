# DS2: Improving Data Efficiency via Curating LLM-Driven Rating Systems


<a href='https://github.com/JlPang863/LLM_data_selection'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2410.10877'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 

[Jinlong Pang](https://jlpang863.github.io/), [Jiaheng Wei](https://sites.google.com/ucsc.edu/jiahengwei), [Ankit Parag Shah](https://ankitshah009.github.io/), [Zhaowei Zhu](https://users.soe.ucsc.edu/~zhaoweizhu/),  [Yaxuan Wang](https://supergirl-os.github.io/), [Chen Qian](https://users.soe.ucsc.edu/~qian/), [Yang Liu](http://www.yliuu.com/), [Yujia Bao](https://www.yujia.io/) and [Wei Wei](http://www.weiwei.one/).

ICLR 2025

University of California, Santa Cruz

<!-- More recent methods have begun to directly leverage the most powerful LLM GPT-4 as data selectors, utilizing their ability to score and filter large-scale datasets with greater precision. 
However, like human annotations, these machine-generated labels (scores) may still be inaccurate or contain LLM-level biases.
Applying these raw labels directly in the data selection process without considering the potential label noise may result in a sub-optimal case.
In this project, we analyze the error patterns in LLM-generated scores and propose a novel data selection pipeline to enhance machine alignment. Our method incorporates label curation and noise reduction techniques over LLM scored data, meanwhile, considers the rareness of the data sample to improve both the accuracy and richness of the selected data. Empirical results demonstrate that our approach not only outperforms existing methods as well as full data training, but also reduces reliance on costly expert-driven models, achieving a more efficient and reliable alignment process. -->

------ 

## 🎉🎉 News 
- [x] [2025.02.01] 👏👏 Accepted by **ICLR 2025**.
- [x] [2024.12.10] 📢📢 Release the [curated dataset](https://huggingface.co/datasets/jlpang888/cured_dataset_gpt_4o_mini).
- [x] [2024.12.01] 🚀🚀 Release the code of **DS2**.

## Brief Introduction
This project is motivated by a common phenomenon that the errors of LLM-generated raw rating scores are widespread and vary significantly across different LLMs. Motivated by this, we introduce DS2, a diversity-aware score curation method for data selection.
method for Data Selection.

![The Overview of Data Selection Pipeline](pipeline_overview.png)

- **Prompt-based LLM Rating**: We generate an initial quality score for each data sample using advanced LLMs.
- **Curated Quality Score Generation**: This step corrects potential rating score errors from the previous step by leveraging the Score Transition Matrix to derive a curated quality score.
- **Long-tail Diversity Score Generation**: We score the diversity of each example by measuring the distance between feature embeddings, identifying samples that fall outside common clusters, which tend to be more distinct.
- **Final Data Selection**:  We prioritize data by first sorting based on the curated scores and then by the long-tail scores. This dual sorting strategy helps with removing poor-quality outliers while ensuring a diverse, high-quality dataset.

------ 

## Dataset preparation

This repository follows the codebase from [TULU](https://github.com/allenai/open-instruct). One can download the evaluation and original training data by running

```bash
# eval data
bash model_finetune/scripts/prepare_eval_data.sh

# train data 
bash model_finetune/scripts/prepare_train_data.sh
```
Our selected evaluation and training data are listed below.

| **Category**         | **Dataset**                                  |
|----------------------|----------------------------------------------|
| **Evaluation Data**   | MMLU, TruthfulQA, GSM, BBH, TydiQA           |
| **Training Data**     | Flan v2, OASST1, WizardLM, Dolly, Stanford Alpaca |

<!-- | **Evaluation Data**              | **Original Training Data**         |
|:--------------------------------:|:----------------------------------:|
| MMLU                             | Flan v2                            |
| TruthfulQA                       | OASST1                             |
| GSM                              | WizardLM                           |
| BBH                              | Dolly                              |
| TydiQA                           | Stanford Alpaca                    | -->

## Environment setup
To run training, evaluation, or inference for finetuned models, you need to install the required packages by running the following command (after installing pytorch):
```
pip install -r requirements.txt
```


## 🚀🚀 Get Started

----- 

## Step 1. LLM-prompt-based rating

In this project, we use three labeling models to generate rating scores, including GPT-4o-mini, Mistral-7B-Instruct-v0.3, LLaMA-3.1-8B-Instruct.  In particular, we can use the GPT API call to generate the model answers by executing the code located in the `LLM_scoring` path: 
```
cd LLM_scoring && bash labeling_datasets_api.sh
``` 
For open-source models such as LLaMA and Mistral, you can submit the jobs via launcher to the cluster, i.e., 
```
cd LLM_scoring && launcher run job_labeling.yaml
``` 
or generate scores locally using 
```
cd LLM_scoring && bash scoring_datasets_local.sh
```



## Step 2. Score curation
Th label curation code base is from [Docta](https://github.com/Docta-ai/docta) in the `./score_curation` path. You can execute the score curation by running
```
cd score_curation && bash diagnose_tulu.sh
```
The corresponding curation report files could be found in the path `./score_curation/results`.



## Step 3. Data selection
Given the existing score curation reports, you can directly use the following jupyter notebooks to do data selection including all baselines: `data_gen_baselines_all.ipynb`. The generated subsets can be further used for LLM instruction tuning. Other selected datasets used for ablation study can be also generated from the following jupyter notebooks located in the `./score_curation` path: `data_gen_score_curation.ipynb` and `data_gen_data_scale.ipynb`. In particular, we use `data_gen_score_curation.ipynb` to generate subsets after curating machine-generated raw scores.


We implement nine baselines consists of Random, Perplexity, KNN, [LESS](https://github.com/princeton-nlp/LESS), Completion_length, Full data, [Alpagasus](https://github.com/Lichang-Chen/AlpaGasus/tree/main) (label-filtered), [DEITA](https://github.com/hkust-nlp/deita) (diversity-filtered), Ours w/o. curation and Ours.



## Step 4. Finetune & Evaluation
Given the selected subsets in the path `model_finetune/selected_data/`, you can use the code base from [TULU](https://github.com/allenai/open-instruct) to finetune base models (Mistral or LLaMA) and then do evaluation.
In particular, you can submit the jobs via launcher under the path `model_finetune/`. For example, you can submit the job by running the code 
```
cd model_finetune/ && launcher run job_pipeline_all.yaml
```
<!-- Models and evaluation results are stored in the [Azure StorageAccount](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F6184c5ce-cd29-4d42-bbcc-0fb06a3f97f1%2FresourceGroups%2FACCLLM%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fafminternshipuksouth/path/jinlong/etag/%220x8DCAC3F12DEAFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None).  -->

Futhermore, we can also execute the code locally, e.g.,  
```
cd model_finetune/ && bash run_pipeline_all.sh
```

You can present the final result by running 
```
python model_finetune/read_results.py
```


## Final results 
The final results of LLM judging compared with human-annotated dataset LIMA can be found in `lima_compare_plot.ipynb`. Moreover, for the tabular results, you can check the `reading_results.ipynb` jupyter notebook.


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