import argparse
import os
import shutil
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

NUM_THREADS = 32
parser = argparse.ArgumentParser("gen all vhacd")
parser.add_argument("--object_name", dest="object_name")
parser.add_argument("--input_dir", dest="input_dir")
parser.add_argument("--output_dir", dest="output_dir")
parser.add_argument("--split_loose", dest="split_merge", action="store_true")
parser.add_argument("--urdf", dest="urdf")


args = parser.parse_args()


def parse_urdf(urdf):
    tree = ET.parse(urdf)
    link_to_vms = dict()
    for link in tree.findall("link"):
        vms = link.findall("visual")
        if len(vms) == 0:
            continue
        link_to_vms[link.attrib["name"]] = [os.path.basename(vm.find("geometry/mesh").attrib["filename"]) for vm in vms]
    return link_to_vms


if not os.path.isdir(args.input_dir):
    raise ValueError("Input directory not found: {}".format(args.input_dir))
    quit()

os.makedirs(args.output_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

if args.split_merge:
    tmp_dir = os.path.join(args.output_dir, "tmp", "split")
    os.makedirs(tmp_dir, exist_ok=True)
    ########################
    # split to loose parts #
    ########################
    cmd = "cd {} && blender -b --python step_2_split.py -- {} {}".format(script_dir, args.input_dir, tmp_dir)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
    input_dir = tmp_dir
else:
    input_dir = args.input_dir

tmp_dir = os.path.join(args.output_dir, "tmp", "vhacd")
os.makedirs(tmp_dir, exist_ok=True)

in_fs = []
out_fs = []
if args.urdf is not None:
    # Generate a collision mesh per link for visual meshes in that link
    assert os.path.isfile(args.urdf)
    link_to_vms = parse_urdf(args.urdf)
    for link_name in link_to_vms:
        tmp_link_dir = os.path.join(tmp_dir, link_name)
        os.makedirs(tmp_link_dir, exist_ok=True)
        # Copy visual meshes within the same link to a separate directory: <tmp_dir>/<link_name>
        for vm in link_to_vms[link_name]:
            shutil.copyfile(os.path.join(input_dir, vm), os.path.join(tmp_link_dir, vm))
        # Merge visual meshes within the same link and save it as <tmp_dir>/<link_name>_vm_cm.obj
        cmd = "cd {} && blender -b --python step_2_merge.py -- {} {} {}".format(
            script_dir, link_name + "_vm", tmp_link_dir, tmp_dir
        )
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        # Input files are in <tmp_dir>
        in_fs.append(os.path.join(tmp_dir, link_name + "_vm_cm.obj"))
        # Output files should be put directly in <args.output_dir>
        out_fs.append(os.path.join(args.output_dir, link_name + "_cm.obj"))
else:
    # Generate a single collision mesh for all the visual meshes
    objs = [o for o in os.listdir(input_dir) if os.path.splitext(o)[1] == ".obj"]
    for o in objs:
        # Input files are in <input_dir>
        in_fs.append(os.path.join(input_dir, o))
        # Output files should be put in <tmp_dir>, will be merged later
        out_fs.append(os.path.join(tmp_dir, o))

print("Inititating V-HACD for {} meshes...".format(len(in_fs)))


def vhacd(cmd):
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)


def vhacd_windows(name_in, name_out):


    p.vhacd(name_in, name_out, "vhacd-log.txt")


threads = []
for in_f, out_f in zip(in_fs, out_fs):
    print(in_f, out_f)
    if sys.platform.startswith("win32"):
        thread = threading.Thread(target=vhacd_windows, args=(in_f, out_f))
    elif sys.platform.startswith("linux"):
        cmd = '../../blender_utils/vhacd --input "{}" --output "{}"'.format(in_f, out_f)
        thread = threading.Thread(target=vhacd, args=(cmd,))
    else:
        print("Unsupported platform")
    thread.start()
    threads.append(thread)

print("Waiting for finishing...")
for thread in threads:
    thread.join()

if args.urdf is not None:
    # A collision mesh for each link was already generated. No need to merge.
    pass
else:
    print("Merging V-HACD...")
    ###########################
    # Merge all V-HACD to one #
    ###########################

    cmd = "cd {} && blender -b --python step_2_merge.py -- {} {} {}".format(
        script_dir, args.object_name, tmp_dir, args.output_dir
    )
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

###########################
# Remove tmp folders      #
###########################
tmp_dir = os.path.join(args.output_dir, "tmp")
cmd = "rm -r {}".format(tmp_dir)
subprocess.call(cmd, shell=True)
