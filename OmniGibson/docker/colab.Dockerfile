FROM stanfordvl/omnigibson:colab-docker

# environment settings
ARG DEBIAN_FRONTEND="noninteractive"
ENV OMNIGIBSON_HEADLESS="1"
ENV OMNIGIBSON_REMOTE_STREAMING="webrtc"

# Fix the JS file to allow for remote streaming on the same port (80)
RUN sed -i "s/49100/80/g" /isaac-sim/extscache/omni.services.streamclient.webrtc-1.3.8/web/js/kit-player.js && \
    sed -i -E 's/IsValidIPv4=.*test\(e\)/IsValidIPv4=function(e){return true/g' /isaac-sim/extscache/omni.services.streamclient.webrtc-1.3.8/web/js/kit-player.js

# Install nginx
RUN apt-get update && apt-get install -y nginx && apt-get clean

# Download the demo dataset and the assets
RUN python -m omnigibson.utils.asset_utils --download_omnigibson_robot_assets --download_behavior_1k_assets --accept_license

# Add the nginx configuration file
ADD docker/nginx.conf /etc/nginx/sites-available/default

CMD nginx && python -m omnigibson.examples.robots.robot_control_example --quickstart