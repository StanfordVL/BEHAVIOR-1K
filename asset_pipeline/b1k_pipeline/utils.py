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
    r"^(?P<mesh_basename>(?P<link_basename>(?P<obj_basename>(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>[LC]-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+))(?:-(?P<link_name>[a-z0-9_]+))?)(?:-(?P<parent_link_name>[a-z0-9_]+)-(?P<joint_type>[RPFAC])-(?P<joint_side>lower|upper))?)(?:-L(?P<light_id>[0-9]+))?(?P<meta_info>-M(?P<meta_type>[a-z]+)(?:_(?P<meta_id>[A-Za-z0-9]+))?(?:_(?P<meta_subid>[0-9]+))?)?(?P<tag>(?:-T[a-z]+)*)$"
)
PORTAL_PATTERN = re.compile(
    r"^portal(-(?P<partial_scene>[A-Za-z0-9_]+)(-(?P<portal_id>\d+))?)?$"
)

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
        return mesh.export(f, resolver=FSResolver(out_fs), file_type=filetype, **kwargs)


# Per-worker-subprocess state. ``_og_initializer`` populates this; tasks
# read it via ``og_context()``. ``None`` in the parent process.
_OG_CONTEXT = None


class _OGContext:
    def __init__(self, clear_kwargs):
        self.clear_kwargs = clear_kwargs
        self.cache = {}


def og_context():
    """Return the per-worker OmniGibson context. Only valid inside a task.

    The returned object exposes ``clear_kwargs`` (the kwargs needed by
    ``og.clear()`` after the URDF importer has swapped the stage) and
    ``cache``, a plain dict that persists across tasks on the same worker.
    """
    if _OG_CONTEXT is None:
        raise RuntimeError("og_context() called outside of an OG worker process")
    return _OG_CONTEXT


def _detect_visible_gpus():
    """Return a list of GPU ids visible to this process.

    Honors a pre-existing ``CUDA_VISIBLE_DEVICES`` if set; otherwise queries
    ``nvidia-smi``. Returns an empty list if neither yields anything.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [g.strip() for g in cvd.split(",") if g.strip()]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=5)
    except Exception:
        return []
    return [str(i) for i, line in enumerate(out.splitlines()) if line.strip()]


def _claim_worker_index(counter_path):
    """Atomically increment a counter file and return the previous value.

    Used by ``_og_initializer`` to assign a stable monotonic index to each
    worker (and re-spawned worker) without needing a shared
    ``multiprocessing.Value`` — those can't be pickled across the loky spawn.
    """
    import fcntl

    flags = os.O_RDWR | os.O_CREAT
    fd = os.open(counter_path, flags, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            existing = os.read(fd, 64).decode().strip()
            idx = int(existing) if existing else 0
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(idx + 1).encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return idx


def _og_initializer(og_macros, counter_path, gpus):
    """Run once per worker subprocess at startup. Boots OG.

    Atomically claims a worker index from a file-based counter and pins this
    worker to one GPU (round-robin over ``gpus``) by setting
    ``CUDA_VISIBLE_DEVICES`` *before* OmniGibson is imported.

    Redirects fd 1/2 to per-worker files in :data:`TMP_DIR` so the OG launch
    log (~thousands of lines) doesn't pollute the parent's stdout.
    """
    worker_idx = _claim_worker_index(counter_path)

    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker_idx % len(gpus)]

    TMP_DIR.mkdir(exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    log_fd = os.open(str(TMP_DIR / f"og-worker-{os.getpid()}.log"), flags)
    err_fd = os.open(str(TMP_DIR / f"og-worker-{os.getpid()}.err"), flags)
    os.dup2(log_fd, 1)
    os.dup2(err_fd, 2)
    os.close(log_fd)
    os.close(err_fd)

    print(
        f"[og-worker {os.getpid()}] worker_idx={worker_idx}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )

    from omnigibson.macros import gm

    for key, value in og_macros.items():
        setattr(gm, key, value)

    import omnigibson as og

    og.launch()
    clear_kwargs = dict(
        gravity=og.sim.gravity,
        physics_dt=og.sim.get_physics_dt(),
        rendering_dt=og.sim.get_rendering_dt(),
        sim_step_dt=og.sim.get_sim_step_dt(),
        viewer_width=og.sim.viewer_width,
        viewer_height=og.sim.viewer_height,
        device=og.sim.device,
    )

    global _OG_CONTEXT
    _OG_CONTEXT = _OGContext(clear_kwargs=clear_kwargs)


def _og_task_wrapper(fn, *args, **kwargs):
    """Worker-side wrapper that calls ``og.clear()`` before invoking ``fn``."""
    import omnigibson as og

    og.clear(**_OG_CONTEXT.clear_kwargs)
    return fn(*args, **kwargs)


def launch_cluster(worker_count, og_macros=None):
    """Launch a loky process pool.

    If ``og_macros`` is provided, each worker subprocess runs
    :func:`_og_initializer` at startup — applying the macros, calling
    ``og.launch()``, and stashing ``og.clear`` kwargs for later use. Workers
    are pinned to GPUs round-robin via ``CUDA_VISIBLE_DEVICES`` (re-pinned
    each time loky respawns one). Use :func:`submit_og_task` to submit jobs
    that should run with a freshly cleared simulator.
    """
    from loky import ProcessPoolExecutor
    from loky.backend.context import get_context

    ctx = get_context("loky")
    if og_macros is None:
        return ProcessPoolExecutor(max_workers=worker_count, context=ctx)

    gpus = _detect_visible_gpus()
    TMP_DIR.mkdir(exist_ok=True)
    counter_path = str(TMP_DIR / f"og-worker-counter-{os.getpid()}")
    # Ensure a fresh counter for this cluster.
    with open(counter_path, "w") as f:
        f.write("0")

    return ProcessPoolExecutor(
        max_workers=worker_count,
        context=ctx,
        initializer=_og_initializer,
        initargs=(dict(og_macros), counter_path, gpus),
    )


def submit_og_task(executor, fn, *args, **kwargs):
    """Submit ``fn`` so that ``og.clear()`` runs on the worker before it."""
    return executor.submit(_og_task_wrapper, fn, *args, **kwargs)
