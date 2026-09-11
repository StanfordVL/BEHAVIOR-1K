"""Scan a serial port for all Dynamixel motors and reboot each one found."""

import argparse
import time

from dynamixel_sdk import PacketHandler, PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS


def scan_and_reboot(port: str, baudrate: int, protocol: float) -> None:
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(protocol)

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open port {port}")
    if not port_handler.setBaudRate(baudrate):
        raise RuntimeError(f"Failed to set baudrate {baudrate}")

    print(f"Scanning {port} at {baudrate} baud (Protocol {protocol})...")

    if protocol == 2.0:
        data_list, result = packet_handler.broadcastPing(port_handler)
        if result != COMM_SUCCESS:
            print(f"Broadcast ping failed: {packet_handler.getTxRxResult(result)}")
            port_handler.closePort()
            return
        else:
            found_ids = sorted(data_list.keys())
    else:
        # Protocol 1.0 has no broadcastPing; ping each ID individually.
        found_ids = []
        for dxl_id in range(1, 254):
            _, result, _ = packet_handler.ping(port_handler, dxl_id)
            if result == COMM_SUCCESS:
                found_ids.append(dxl_id)

    if not found_ids:
        print("No motors found.")
        port_handler.closePort()
        return

    print(f"Found {len(found_ids)} motor(s): IDs {found_ids}")

    for dxl_id in found_ids:
        result, error = packet_handler.reboot(port_handler, dxl_id)
        if result == COMM_SUCCESS:
            print(f"  ID {dxl_id}: rebooted")
        else:
            print(f"  ID {dxl_id}: reboot failed — {packet_handler.getTxRxResult(result)}")
        time.sleep(0.3)  # motors need a moment to come back up after reboot

    print("Done.")
    port_handler.closePort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and reboot all Dynamixel motors on a port.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=2000000, help="Baud rate (default: 2000000)")
    parser.add_argument("--protocol", type=float, default=2.0, choices=[1.0, 2.0], help="Protocol version (default: 2.0)")
    args = parser.parse_args()

    scan_and_reboot(args.port, args.baudrate, args.protocol)
