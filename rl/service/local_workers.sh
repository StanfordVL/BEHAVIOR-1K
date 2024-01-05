
# Loop from 1 to 12
for i in {1..12}
do
    # Pick a port using the array index
    BASE_PORT=51000
    WORKER_PORT=$((BASE_PORT + i))

    # Pick which GPU to use
    if [ $((foo%2)) -eq 0 ];
    then
        USE_GPU=0
    else
        USE_GPU=1
    fi

    # Run the command inside a screen
    OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=$USE_GPU screen -dmS worker$i python omni_grpc_worker.py 127.0.0.1:50051 $WORKER_PORT
done