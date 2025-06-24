from asyncio import subprocess
import subprocess
import sys
import xmlrpc.client

# TODO: Remove this and call it with python -m when ready to rerun everything.
sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")
import b1k_pipeline.utils

USE_RPC = False

THREEDSMAX_PATH = r"C:\Program Files\Autodesk\3ds Max 2022\3dsmaxbatch.exe"


def main():
    scene_file = (b1k_pipeline.utils.PIPELINE_ROOT / sys.argv[1]).absolute()
    assert (
        scene_file.exists() and scene_file.suffix == ".max"
    ), f"Can't find {scene_file}."

    script_file = (b1k_pipeline.utils.PIPELINE_ROOT / sys.argv[2]).absolute()
    assert (
        script_file.exists() and script_file.suffix == ".py"
    ), f"Can't find {script_file}."

    args = [x.split("=") for x in sys.argv[3:]]
    parsed_args = {key: value for key, value in args}
    assert all(key and value for key, value in parsed_args.items())

    if USE_RPC:
        with xmlrpc.client.ServerProxy("http://localhost:8000/") as proxy:
            return proxy.run_script(str(scene_file), str(script_file), args)
    else:
        max_args = [
            ["-mxsString", f"{key}:{value}"] for key, value in parsed_args.items()
        ]
        cmd = [THREEDSMAX_PATH, str(script_file), "-sceneFile", str(scene_file)]
        if max_args:
            cmd += [x for args in max_args for x in args]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
