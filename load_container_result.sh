


SEED=44 #44
cluster_root_path="data_scale_result_seed${SEED}" ## . for local

# cluster_root_path="data_scale_result" ## . for local


AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/${cluster_root_path}/results/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
azcopy copy "https://afminternshipuksouth.blob.core.windows.net/jinlong/${cluster_root_path}/results/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D" "data_scale_result_seed${SEED}/"  --recursive
