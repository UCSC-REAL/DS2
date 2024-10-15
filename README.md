# LLM Data Selection Pipeline
More recent methods have begun to directly leverage the most powerful LLM GPT-4 as data selectors, utilizing their ability to score and filter large-scale datasets with greater precision. 
However, like human annotations, these machine-generated labels (scores) may still be inaccurate or contain LLM-level biases.
Applying these raw labels directly in the data selection process without considering the potential label noise may result in a sub-optimal case.
In this project, we analyze the error patterns in LLM-generated scores and propose a novel data selection pipeline to enhance machine alignment. Our method incorporates label curation and noise reduction techniques over LLM scored data, meanwhile, considers the rareness of the data sample to improve both the accuracy and richness of the selected data. Empirical results demonstrate that our approach not only outperforms existing methods as well as full吃的 data training, but also reduces reliance on costly expert-driven models, achieving a more efficient and reliable alignment process.

# 1. LLM-prompt-based Rating

In this project, we use three labeling models to generate rating scores, including GPT-4o-mini, Mistral-7B-Instruct-v0.3, LLaMA-3.1-8B-Instruct. The code is located in the path: ./open-instruct/LLM Scoring/. In particular, we use the labeling_datasets_api.sh to call the GPT API to generate the model answers, i.e., `LLM Scoring/labeling_datasets_api.sh`. For LLaMA and Mistral, one can submit the jobs via launcher to the cluster, i.e., `launcher run job_labeling.yaml`.



# 2. Label Curation Method
Th label curation code base is from [Docta](https://github.com/Docta-ai/docta) in the `./labeling` path. One can execute the label curation by running
```
bash diagnose_tulu.sh
```
One can check the label curation report file in the path `./labeling/results`.

# 3. Data Selection including baselines
Given the existing label curation reports, one can directly use the following jupyter notebooks to do data selection including all baselines: `./labeling/new_dataset_all.ipynb`. Other selected datasets used for ablation study can be also generated from the following jupyter notebooks: `./labeling/new_dataset_label_curation.ipynb` and `./labeling/new_dataset_data_scale.ipynb`



# 4. Finetune & Evaluation
Given the selected subsets, one can use the code base from [TULU](https://github.com/allenai/open-instruct) to finetune and do evaluation.
In particular, one can submit the jobs via launcher under the path `./open-instruct/model_finetune_cluster/`. For example, one can submit the job by running the code `launcher run job_pipeline_all.yaml`. Models and evaluation results are stored in the [Azure StorageAccount](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F6184c5ce-cd29-4d42-bbcc-0fb06a3f97f1%2FresourceGroups%2FACCLLM%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fafminternshipuksouth/path/jinlong/etag/%220x8DCAC3F12DEAFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None)  One can present the final result by running `python ./open-instruct/model_finetune_cluster/read_results.py`.


