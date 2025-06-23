from xmlrpc.server import SimpleXMLRPCServer

import pymxs

rt = pymxs.runtime


def run_script(scene_file, script_path, args):
    # Load the file.
    rt.resetMaxFile(rt.name("noPrompt"))
    assert rt.loadMaxFile(
        scene_file, useFileUnits=False, quiet=True
    ), f"Could not load {scene_file}"

    # Set the arguments.
    rt.maxops.mxsCmdLineArgs = rt.Dictionary()
    for arg_key, arg_value in args:
        rt.maxops.mxsCmdLineArgs[rt.name(arg_key)] = arg_value

    # Execute the script
    rt.python.ExecuteFile(script_path)

    return True


def main():
    server = SimpleXMLRPCServer(("localhost", 8000))
    print("Listening on port 8000...")
    server.register_function(run_script, "run_script")
    server.serve_forever()


if __name__ == "__main__":
    main()
