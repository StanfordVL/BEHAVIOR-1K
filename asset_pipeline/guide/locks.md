# Locking Files

## Problem
* Our work on the dataset involves 10+ people editing ~80 processed.max files concurrently.
* This means that there is a high likelihood that multiple people will edit the same file if we don't pay attention to avoiding it.
* Since the files are not text files but instead 3ds Max binaries, if multiple people edit the same file, **there is no easy way to merge their changes.**
* This means we'll have to pick one version or the other, and the unpicked version's author will have to redo their changes.

## Solution
* We use a [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1p5SA2Pt44UHcMZsT3IeOHEVPb8TSPkFro_i8bvWodQA/edit?gid=1640497008#gid=1640497008) to grab a "lock" on a file to indicate that you **exclusively** have the right to edit it.
* Only one person may hold a file's lock at once. If a file is already locked by somebody you cannot grab its lock.
* Every can **only** edit files they hold the lock for.

## Instruction
When you are going to edit a file, follow the below steps:

1. Check that the file is not currently locked. If it is, and you need access, ping the lock holder to get them to merge their changes and unlock the file.
2. Put your GitHub username on column C (Github Acc Holding Lock) of the row corresponding to the file.
3. On your computer, `git checkout main`, `git pull` to get the latest DVC pointers.
4. Create a new branch for your work: `git checkout -b your_feature`. Or if you already have a branch, switch to the branch and merge `main` into your branch:  `git checkout your_branch; git merge main`.
5. Run `dvc pull` to check out the latest version of all the files from DVC.
6. Unprotect the file you are going to work on: `dvc unprotect cad/x/x/processed.max`
7. Do the work on the file and save it. **Again: ONLY work on files you hold the lock for!**
8. Add the file to DVC: `dvc add cad/x/x/processed.max`
9. Commit your change `git commit -m "what did you change?"`
10. Push your change to both DVC and Git: `dvc push` and then `git push`
11. Create a PR on the repository. Copy the link to the PR.
12. **Only if you are not going to continue working on the file**, e.g. if your PR includes ALL the changes you are going to make to that file, paste the PR link on the "Unlock on PR" column (Column D) of the spreadsheet. **For every file that you have NOT done ANY work on (e.g. you DON'T have some partial work that you already started but haven't included in your PR), make sure that you RELEASE the file lock by deleting your name so other people can work on it during your off hours. You can always grab the lock again in the morning!**
13. Cem will soon merge your PR, and after merging the PR, if you put the PR number on the column indicating you're done with the file, Cem will remove your name from the lock column, unlocking the file. **That means, before doing any further work on that file, you MUST repeat steps 1-5**!

**Since multiple people will be editing the same files in quick succession, you need to make sure you have the most recent version of the file by doing a git/dvc merge+pull from `main` EVERY TIME you start work after obtaining a file lock!** Just obtaining the lock is not sufficient since if you don't pull you will not have the most recent changes.