import os
import pathlib
import re
import sys

import fs.path
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs.zipfs import ZipFS
import numpy as np
import trimesh.resolvers
import yaml
import subprocess

PIPELINE_ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP_DIR = PIPELINE_ROOT / "tmp"
PARAMS_FILE = PIPELINE_ROOT / "params.yaml"
NAME_PATTERN = re.compile(
    r"^(?P<mesh_basename>(?P<link_basename>(?P<obj_basename>(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>[LC]-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+))(?:-(?P<link_name>[a-z0-9_]+))?)(?:-(?P<parent_link_name>[a-z0-9_]+)-(?P<joint_type>[RPCFA])-(?P<joint_side>lower|upper))?)(?:-L(?P<light_id>[0-9]+))?(?P<meta_info>-M(?P<meta_type>[a-z]+)(?:_(?P<meta_id>[A-Za-z0-9]+))?(?:_(?P<meta_subid>[0-9]+))?)?(?P<tag>(?:-T[a-z]+)*)$"
)
PORTAL_PATTERN = re.compile(
    r"^portal(-(?P<partial_scene>[A-Za-z0-9_]+)(-(?P<portal_id>\d+))?)?$"
)
CLUSTER_MODE = "enroot"  # one of "docker", "slurm", "enroot"

params = yaml.load(open(PARAMS_FILE, "r"), Loader=yaml.SafeLoader)


def parse_name(name):
    return NAME_PATTERN.fullmatch(name)


def parse_portal_name(name):
    return PORTAL_PATTERN.fullmatch(name)


def get_targets(target_type):
    return list(params[target_type])


class WriteOnly7ZipFS(TempFS):
    """
    A write-only filesystem that stores data in a temporary directory,
    and upon closing, compresses it using 7zip into a final zip archive.
    """

    def __init__(self, zip_path, temp_fs=None, **kwargs):
        """
        Initialize the write-only 7zip-backed TempFS.

        :param zip_path: Destination path for the resulting .zip file.
        :param kwargs: Other arguments passed to TempFS.
        """
        self._temp_fs = (
            temp_fs  # We keep this pointer to avoid deallocation of the tempfs
        )
        if self._temp_fs is not None:
            kwargs["temp_dir"] = self._temp_fs.getsyspath("/")
        super().__init__(**kwargs)
        self._zip_path = os.path.abspath(zip_path)
        self._closed = False

    def close(self):
        """
        On close, compress the entire TempFS contents into a zip file using 7z.
        """
        if not self.isclosed():
            temp_path = self.getsyspath("/")
            try:
                sevenzip_cmd = (
                    str((PIPELINE_ROOT / "7za.exe").resolve())
                    if sys.platform == "win32"
                    else "7z"
                )  # TODO: Make this work on Windows too
                subprocess.run(
                    [sevenzip_cmd, "a", self._zip_path, "."],
                    cwd=temp_path,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if not os.path.exists(self._zip_path):
                    raise ValueError("7zip did not produce expected zip file.")
            except subprocess.CalledProcessError as e:
                raise ValueError(f"7zip failed: {e.stderr.decode().strip()}") from e
            finally:
                super().close()  # Flush and close TempFS


class PipelineFS(OSFS):
    def __init__(self) -> None:
        super().__init__(PIPELINE_ROOT)

    def pipeline_output(self):
        return self.opendir("artifacts/pipeline")

    def target(self, target):
        return self.opendir(fs.path.join("cad", target))

    def target_output(self, target):
        return self.target(target).makedir("artifacts", recreate=True)


def ParallelZipFS(name, write=False, temp_fs=None):
    if not temp_fs:
        TMP_DIR.mkdir(exist_ok=True)
        temp_fs = TempFS(temp_dir=str(TMP_DIR))
    zip_filename = PIPELINE_ROOT / "artifacts/parallels" / name
    if not write:
        return ZipFS(zip_filename, write=False, temp_fs=temp_fs)
    else:
        return WriteOnly7ZipFS(zip_filename, temp_fs=temp_fs)


def mat2arr(mat, dtype=np.float32):
    return np.array(
        [
            [mat.row1.x, mat.row1.y, mat.row1.z],
            [mat.row2.x, mat.row2.y, mat.row2.z],
            [mat.row3.x, mat.row3.y, mat.row3.z],
            [mat.row4.x, mat.row4.y, mat.row4.z],
        ],
        dtype=dtype,
    )


class FSResolver(trimesh.resolvers.Resolver):
    """
    Resolve files from a source path on the file system.
    """

    def __init__(self, fs):
        self._fs = fs

    def namespaced(self, namespace):
        return FSResolver(self._fs.opendir(namespace))

    def get(self, name):
        """
        Get an asset.

        Parameters
        -------------
        name : str
          Name of the asset

        Returns
        ------------
        data : bytes
          Loaded data from asset
        """
        # load the file by path name
        with self._fs.open(name.strip(), "rb") as f:
            data = f.read()
        return data

    def keys(self):
        """
        List all files available to be loaded.

        Yields
        -----------
        name : str
          Name of a file which can be accessed.
        """
        yield from self._fs.walk.files()

    def write(self, name, data):
        """
        Write an asset to a file path.

        Parameters
        -----------
        name : str
          Name of the file to write
        data : str or bytes
          Data to write to the file
        """
        # write files to path name
        with self._fs.open(name.strip(), "wb") as f:
            # handle encodings correctly for str/bytes
            trimesh.util.write_encoded(file_obj=f, stuff=data)


def load_points(fs, name):
    data = fs.readtext(name)
    points = []

    for line in data.split("\n"):
        if not line.startswith("v "):
            continue
        x, y, z = [float(x) for x in line.replace("v ", "").split()]
        points.append([x, y, z])

    return np.array(points)


def load_mesh(fs, name, **kwargs):
    with fs.open(name, "rb") as f:
        return trimesh.load(f, resolver=FSResolver(fs), file_type="obj", **kwargs)


def save_mesh(mesh, out_fs, name, **kwargs):
    with out_fs.open(name, "wb") as f:
        filetype = fs.path.splitext(name)[1][1:]  # Get file extension without dot
        return mesh.export(f, resolver=FSResolver(out_fs), file_type=filetype, include_normals=False, include_color=False, **kwargs)


WORKER_APPDATA_ROOT = TMP_DIR / "worker-appdata"


def _detect_gpus():
    """List GPU device ids to round-robin across.

    If ``CUDA_VISIBLE_DEVICES`` is already set in the parent env, respect
    that restriction. Otherwise enumerate via ``nvidia-smi``. Falls back to
    ``["0"]`` if neither is available.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return [g.strip() for g in visible.split(",") if g.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        return gpus or ["0"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["0"]


def _og_pool_worker_init(counter, gpus):
    """ProcessPoolExecutor initializer: pin this worker process to one GPU.

    Each worker atomically claims the next index from a shared counter and
    sets ``CUDA_VISIBLE_DEVICES`` to ``gpus[idx % len(gpus)]``. Subprocesses
    spawned by this worker (via ``worker_subprocess_env``) inherit that env
    var, so all OmniGibson work the worker triggers lands on the same GPU.
    Within the visible set, ``OMNIGIBSON_GPU_ID`` is always ``"0"``.
    """
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    gpu = gpus[idx % len(gpus)]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def make_og_pool_executor(worker_count, gpus=None):
    """Build a ``ProcessPoolExecutor`` that pins workers round-robin to GPUs.

    Workers run a tiny initializer that grabs an index from a shared atomic
    counter and writes ``CUDA_VISIBLE_DEVICES`` for the worker process; any
    subprocesses spawned by the worker then inherit that pinning through
    :func:`worker_subprocess_env` (which copies ``os.environ`` and tacks on
    a per-pid ``OMNIGIBSON_APPDATA_PATH``). The mapping is deterministic in
    the worker→GPU direction (worker N → GPU N % len(gpus)), so an
    even-allocation invariant holds even if individual tasks are scheduled
    on whichever worker is free.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Value

    if gpus is None:
        gpus = _detect_gpus()
    counter = Value("i", 0)
    return ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_og_pool_worker_init,
        initargs=(counter, gpus),
    )


def worker_subprocess_env():
    """Build the env dict to hand to ``subprocess.Popen`` from a pool worker.

    Each pool worker process gets its own ``OMNIGIBSON_APPDATA_PATH`` under
    ``tmp/worker-appdata/pid-<pid>``. OmniGibson reads this env var in
    ``macros.py`` and routes both ``--portable-root`` and
    ``--/app/tokens/omni_global_cache`` to it, so concurrent OG subprocesses
    no longer fight over a single shared ``texturecache`` (which was
    accumulating to ~200GB and triggering ``LocalDataStore`` segfaults on
    block-load failures). Sequential subprocess invocations from the same
    pool worker still reuse the cache, since the path is keyed on the
    worker PID, not per-task.

    ``CUDA_VISIBLE_DEVICES`` and ``OMNIGIBSON_GPU_ID`` are *not* set here —
    they're already set on the worker process by :func:`_og_pool_worker_init`,
    and ``dict(os.environ)`` propagates that pinning to the subprocess.
    """
    appdata = WORKER_APPDATA_ROOT / f"pid-{os.getpid()}"
    appdata.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMNIGIBSON_APPDATA_PATH"] = str(appdata.absolute())
    return env
