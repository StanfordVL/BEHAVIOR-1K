
# Loop from 1 to 12
for i in {1..12}
do
    # Run the command inside a screen
    screen -dmS worker$i ./local_enroot_worker.sh cgokmen-lambda.stanford.edu:50051 $i
done