import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

pairs = [
  # ("/s3/profiles/fb-fair-siro-fsx-dra/cgokmen/behavior-data2", "/checkpoint/clear/cgokmen/behavior-data2"),
  # ("/s3/profiles/fb-fair-siro-fsx-dra/cgokmen/og-materials", "/checkpoint/clear/cgokmen/og-materials"),
  # ("/s3/profiles/fb-fair-siro-fsx-dra/cgokmen/procthor", "/checkpoint/clear/cgokmen/procthor"),
  ("/s3/profiles/fb-fair-siro-fsx-dra/cgokmen/habitat-data", "/checkpoint/clear/cgokmen/habitat-data"),
]

def copy_file(src, dst):
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(src, dst)

def main():
  with ThreadPoolExecutor() as executor:
    futures = []

    with tqdm(desc="Queueing files") as pbar:
      for from_dir, to_dir in pairs:
        from_dir = pathlib.Path(from_dir)
        to_dir = pathlib.Path(to_dir)
        for dirpath, _, files in from_dir.walk():
          for file in files:
            file = pathlib.Path(dirpath) / file
            target_path = to_dir / file.relative_to(from_dir)
            futures.append(executor.submit(copy_file, file, target_path))
            pbar.update(1)

    for future in tqdm(as_completed(futures), total=len(futures), desc="Copying files"):
      future.result()

if __name__ == "__main__":
  main()