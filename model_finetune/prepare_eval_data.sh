mkdir -p data/downloads
mkdir -p data/eval

# MMLU dataset
wget -O data/downloads/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/downloads/mmlu_data
tar -xvf data/downloads/mmlu_data.tar -C data/downloads/mmlu_data
mv data/downloads/mmlu_data/data data/eval/mmlu && rm -r data/downloads/mmlu_data data/downloads/mmlu_data.tar


# Big-Bench-Hard dataset
wget -O data/downloads/bbh_data.zip https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip
mkdir -p data/downloads/bbh
unzip data/downloads/bbh_data.zip -d data/downloads/bbh
mv data/downloads/bbh/BIG-Bench-Hard-main/ data/eval/bbh && rm -r data/downloads/bbh data/downloads/bbh_data.zip


# TyDiQA-GoldP dataset
mkdir -p data/eval/tydiqa
wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json
wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json

# GSM dataset
wget -P data/eval/gsm/ https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl


# TruthfulQA
wget -P data/eval/truthfulqa https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv

