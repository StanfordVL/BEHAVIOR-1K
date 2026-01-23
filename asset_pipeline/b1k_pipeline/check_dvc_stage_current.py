import sys
from dvc.repo import Repo, lock_repo


def main():
    (stage_name,) = sys.argv[1:]
    repo = Repo(".")
    stage_infos = repo.stage.collect_granular(stage_name, with_deps=True)
    stages = [si.stage for si in stage_infos]
    assert len(stages) > 0, f"Stage {stage_name} not found!"

    up_to_date = True
    with lock_repo(repo):
        for s in stages:
            if s.changed(allow_missing=True):
                up_to_date = False
                print(f"Stage {s} is out-of-date.")

    if up_to_date:
        print(stage_name, "is up-to-date.")
        sys.exit(0)
    else:
        print(stage_name, "is NOT up-to-date. Please reproduce.")
        sys.exit(1)


if __name__ == "__main__":
    main()
