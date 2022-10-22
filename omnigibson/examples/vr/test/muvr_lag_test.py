""" Test script to measure lag and figure out packet arrival times.

Usage: python muvr_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
"""

import argparse
import time

from omnigibson.utils.muvr_utils import IGVRTestClient, IGVRTestServer

# Client and server settings
SERVER_FPS = 30.0
# Note: size is measured in number of floats
SERVER_PACKET_SIZE = 100000
CLIENT_FPS = 30.0
CLIENT_PACKET_SIZE = 100000


def run_lag_test(mode="server", host="localhost", port="7500"):
    """
    Sets up the OmniGibson environment that will be used by both server and client
    """
    print("INFO: Running MUVR {} at {}:{}".format(mode, host, port))
    # This function only runs if mode is one of server or client, so setting this bool is safe
    is_server = mode == "server"

    # Setup client/server
    if is_server:
        vr_server = IGVRTestServer(localaddr=(host, port))
        vr_server.set_packet_size(SERVER_PACKET_SIZE)
    else:
        vr_client = IGVRTestClient(host, port)
        vr_client.set_packet_size(CLIENT_PACKET_SIZE)

    # Main networking loop
    while True:
        frame_start = time.time()
        if is_server:
            # Simulate server FPS
            time.sleep(1 / SERVER_FPS)

            # Generate and send latest rendering data to client
            vr_server.gen_packet()
            vr_server.send_packet()
            vr_server.Refresh()
        else:
            # Simulate client FPS
            time.sleep(1 / CLIENT_FPS)

            # Generate and send latest VR data to server
            vr_client.gen_packet()
            vr_client.send_packet()
            vr_client.Refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-user VR demo that can be run in server and client mode.")
    parser.add_argument("--mode", default="server", help="Mode to run in: either server or client")
    parser.add_argument("--host", default="localhost", help="Host to connect to - eg. localhost or an IP address")
    parser.add_argument("--port", default="7500", help="Port to connect to - eg. 7500")
    args = parser.parse_args()
    try:
        port = int(args.port)
    except ValueError as ve:
        print("ERROR: Non-integer port-number supplied!")
    if args.mode in ["server", "client"]:
        run_lag_test(mode=args.mode, host=args.host, port=port)
    else:
        print("ERROR: mode {} is not supported!".format(args.mode))
