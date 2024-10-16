# Intern Project: Improving Data Efficiency via Curating LLM-Driven Rating Systems 

## LLM Data Selection Pipeline
More recent methods have begun to directly leverage the most powerful LLM GPT-4 as data selectors, utilizing their ability to score and filter large-scale datasets with greater precision. 
However, like human annotations, these machine-generated labels (scores) may still be inaccurate or contain LLM-level biases.
Applying these raw labels directly in the data selection process without considering the potential label noise may result in a sub-optimal case.
In this project, we analyze the error patterns in LLM-generated scores and propose a novel data selection pipeline to enhance machine alignment. Our method incorporates label curation and noise reduction techniques over LLM scored data, meanwhile, considers the rareness of the data sample to improve both the accuracy and richness of the selected data. Empirical results demonstrate that our approach not only outperforms existing methods as well as full吃的 data training, but also reduces reliance on costly expert-driven models, achieving a more efficient and reliable alignment process.

---

## Step 1. LLM-prompt-based Rating

In this project, we use three labeling models to generate rating scores, including GPT-4o-mini, Mistral-7B-Instruct-v0.3, LLaMA-3.1-8B-Instruct.  In particular, we can use the GPT API call to generate the model answers by executing the code `LLM_Scoring/labeling_datasets_api.sh`. For open-source models such as LLaMA and Mistral, one can submit the jobs via launcher to the cluster, i.e., `launcher run job_labeling.yaml` or generate scores locally using `scoring_datasets_local.sh`.

---

## Step 2. Score Curation Method
Th label curation code base is from [Docta](https://github.com/Docta-ai/docta) in the `./score_curation` path. One can execute the score curation by running
```
bash diagnose_tulu.sh
```
The corresponding curation report files could be found in the path `./score_curation/results`.

---

## Step 3. Data Selection including baselines
Given the existing score curation reports, one can directly use the following jupyter notebooks to do data selection including all baselines: `new_dataset_all.ipynb`. The generated subsets can be further used for LLM instruction tuning. Other selected datasets used for ablation study can be also generated from the following jupyter notebooks located in `./score_curation`: `new_dataset_label_curation.ipynb` and `new_dataset_data_scale.ipynb`
We implement nine baselines consists of random, perplexity, knn, [less](https://github.com/princeton-nlp/LESS), completion_length, full data, [alpagasus](https://github.com/Lichang-Chen/AlpaGasus/tree/main) (label-filtered), [deita](https://github.com/hkust-nlp/deita) (diversity-filtered), ours w/o. curation and ours.
In particular, we use `new_dataset_score_curation.ipynb` to generate subset after curating machine-generated raw scores.

---

## Step 4. Finetune & Evaluation
Given the selected subsets in the path `model_finetune_cluster/new_train_data`, one can use the code base from [TULU](https://github.com/allenai/open-instruct) to finetune base models (Mistral or LLaMA) and then do evaluation.
In particular, one can submit the jobs via launcher under the path `model_finetune_cluster/`. For example, one can submit the job by running the code `launcher run job_pipeline_all.yaml`. Models and evaluation results are stored in the [Azure StorageAccount](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F6184c5ce-cd29-4d42-bbcc-0fb06a3f97f1%2FresourceGroups%2FACCLLM%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fafminternshipuksouth/path/jinlong/etag/%220x8DCAC3F12DEAFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None)  One can present the final result by running `python model_finetune_cluster/read_results.py`.

---

## Final Results 
The final results of LLM judging compared with human-annotated dataset LIMA can be found in `lima_plot.ipynb`. Moreover, for the tabular results, one can check the `reading_results.ipynb` jupyter notebook.
