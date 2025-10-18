#!/bin/bash

# Number of times to run the Python script
num_runs=10

# Loop to run the Python script multiple times
for ((i=1; i<=num_runs; i++))
do
  echo "Run #$i"
  python generate_random_poses_2.py 100 test.json
done

