import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile

from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401

ENV_TFNETBUILDER_HOME = 'TFNETBUILDER_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

try:
    from tqdm.auto import tqdm  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):  # type: ignore

            def __init__(self, total=None, disable=False,
                         unit=None, unit_scale=None, unit_divisor=None):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
                sys.stderr.flush()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write('\n')


def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "netbuilder hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def _get_tf_net_checkpoints_home():
    net_home = os.path.expanduser(
        os.getenv(ENV_TFNETBUILDER_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                         DEFAULT_CACHE_DIR), 'tf_netbuilder')))
    return net_home


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def find_fname(disposition_item, key):
    i1 = disposition_item.find(key)
    if i1 > -1:
        fname_block = disposition_item[i1 + len(key):]
        fname_reg = re.compile(r'([a-z0-9\_\-]*\.[a-z]{3})').search(fname_block)
        filename = fname_reg.group(0)
        return filename
    return None


def download_file(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    """
    Downloads and caches a file. ZIP files will be uncompressed. Returns a path to a local file or folder where the
    zip file was unzipped
    """
    if model_dir is None:
        hub_dir = _get_tf_net_checkpoints_home()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    if file_name is not None:
        filename = file_name
    else:
        req = Request(url, headers={"User-Agent": "models hub"})
        u = urlopen(req)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            disp = meta.getheaders("Content-Disposition")[0]
        else:
            disp = meta.get_all("Content-Disposition")[0]
        server_file_name = find_fname(disp, key="filename*=")
        if server_file_name is None:
            server_file_name = find_fname(disp, key="filename=")

        assert len(server_file_name) > 0

        filename = server_file_name

    downloaded_file = os.path.join(model_dir, filename)
    nm, ext = os.path.splitext(filename)
    maybe_zip = ext == '.zip'

    if maybe_zip:
        cached_file = os.path.join(os.path.dirname(downloaded_file), nm)
    else:
        cached_file = downloaded_file

    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, downloaded_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None

        _download_url_to_file(url, downloaded_file, hash_prefix, progress=progress)

        if zipfile.is_zipfile(downloaded_file):
            try:
                with zipfile.ZipFile(downloaded_file) as cached_zipfile:
                    cached_zipfile.extractall(model_dir)
                    _remove_if_exists(downloaded_file)
                    assert os.path.exists(cached_file)
            except Exception:
                _remove_if_exists(cached_file)
                raise

    return cached_file


def download_checkpoint(url, model_dir=None, progress=True, check_hash=False, file_name=None, checkpoint_file=None):
    path = download_file(url, model_dir, progress, check_hash, file_name)
    if checkpoint_file:
        return os.path.join(path, checkpoint_file)
    else:
        chk_folder = os.path.basename(path)
        return os.path.join(path, chk_folder)
