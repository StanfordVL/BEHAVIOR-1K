import gspread
import os
import subprocess
import time
from google.oauth2.service_account import Credentials
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


user = os.environ.get("USER")
MAX_JOBS = 100  # Maximum number of jobs allowed

# 1. Set up credentials and authorize
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = "/afs/cs.stanford.edu/u/{}/google_credentials.json".format(user)

credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(credentials)


spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")

# Get number of running or pending jobs for the current user
cmd = (
    "sacct --format=JobID,State --user={} --state=RUNNING,PENDING --noheader "
    "| awk '$2 ~ /RUNNING|PENDING/ {{ split($1, a, \".\"); print a[1] }}' "
    "| sort -u "
    "| wc -l"
).format(user)
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
running_jobs = int(result.stdout.strip())
if running_jobs >= MAX_JOBS:
    print(f"SLURM job limit reached: {running_jobs}. Exiting...")
    exit(0)

num_job_scheduled = 0

task_list = list(TASK_NAMES_TO_INDICES.keys())
worksheets = spreadsheet.worksheets()
for ws in worksheets:
    if ws.title in task_list:
        # Iterate through all the rows, find the unprocessed ones
        for row in ws.get_all_values():
            if row and row[1].strip().lower() == "unprocessed":
                # TODO: get raw data from lightwheel
                # TODO: extract data and get file id
                file_id = ""
                num_job_scheduled += 1
                ws.update_cell(ws.find(row[0]).row, 2, "processing")
                ws.update_cell(ws.find(row[0]).row, 3, user)
                ws.update_cell(ws.find(row[0]).row, 4, time.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"Scheduling job for file ID: {file_id} in task: {ws.title}")
                cmd = (
                    "cd /vision/u/{}/BEHAVIOR-1K && "
                    'sbatch OmniGibson/omnigibson/learning/scripts/replay_data.sbatch.sh "/vision/u/{}/data/behavior/raw/{}/{}"'
                ).format(user, user, ws.title, file_id)
                # Run the command non-blocking
                subprocess.Popen(cmd, shell=True, text=True)
                if num_job_scheduled + running_jobs >= MAX_JOBS:
                    print(f"Reached job limit after processing {num_job_scheduled} unprocessed files. Exiting...")
                    exit(0)
