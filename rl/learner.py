import argparse
import subprocess
import uuid
from dask.distributed import Client

def run_rollouts(policy_uuid, max_rollouts_per_worker):
    policy_path = './policies/policy_{}.pkl'.format(policy_uuid)
    rollouts_uuid = uuid.uuid4().hex
    rollouts_path = './rollouts/rollouts_{}.hdf5'.format(rollouts_uuid)

    cmd = " ".join(['OMNIGIBSON_HEADLESS=1', 'python', '-m', "learner", policy_path, rollouts_path, max_rollouts_per_worker])
    subprocess.call(cmd, shell=True)

    return rollouts_path

def main(scheduler_route, num_workers, max_rollouts):
    c = Client(scheduler_route)
    while True:
        # read from your current set of rollouts

        # do some learning

        # generate UUID for new policy
        uuid = uuid.uuid4().hex

        # save policy to disk
        with open('./policies/policy_{}.pkl'.format(uuid), 'wb') as f:
            pass

        max_rollouts_per_worker = round(max_rollouts / num_workers)

        # now call the runner with the new policy
        futures = [c.submit(run_rollouts, uuid, max_rollouts_per_worker) for _ in range(num_workers)]
        
        # wait for all the futures
        rollout_files = [f.result() for f in futures]

        # repeat.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run learner")
    parser.add_argument("scheduler_route")
    parser.add_argument("num_workers")
    parser.add_argument("max_rollouts")
    
    args = parser.parse_args()
    main(args.scheduler_route, args.num_workers, args.max_rollouts)
