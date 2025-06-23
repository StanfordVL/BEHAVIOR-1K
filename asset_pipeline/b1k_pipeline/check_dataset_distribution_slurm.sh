sinfo -p viscam -o "%N,%n" -h | \
  sed s/,.*//g | \
  xargs -L1 -I{} \
    sbatch \
      --account=viscam --partition=viscam --nodelist={} --mem=2G --cpus-per-task=1 \
      --wrap 'python -c '\''import os; print(os.path.exists("/scr-ssd/og-data-1-0-0"))'\'' > /cvgl2/u/cgokmen/dataset-check/${SLURM_NODELIST}.txt'