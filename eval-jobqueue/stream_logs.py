import sys
import os
import google.cloud.logging

user = os.getenv("USER")
job = os.environ.get("SLURM_JOB_ID", "")

# Setup the client (automatically looks for credentials)
client = google.cloud.logging.Client()
logger = client.logger(
    sys.argv[1] if len(sys.argv) > 1 else "slurm"
)  # This will be the log name in GCP console


def main():
    # Batching is handled automatically by the library for performance
    print("Streaming logs to Google Cloud...")
    for line in sys.stdin:
        # .rstrip() removes the double newline
        logger.log_text(line.rstrip(), labels={"job": job, "user": user})


if __name__ == "__main__":
    main()
