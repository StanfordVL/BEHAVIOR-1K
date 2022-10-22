export OMNIGIBSON_DATA_PATH=/path/to/omnomnigibson_data
# To run with GUI: edit macros.py and add -v /tmp/.X11-unix:/tmp/.X11-unix to container launch command
docker run -e DISPLAY -v $OMNIGIBSON_DATA_PATH:/OmniGibson3/omnigibson/data --rm -it omnomnigibson
