"""Optional Triton cache manager for Landlock ABI 1 environments.

Uses Triton's public TRITON_CACHE_MANAGER extension point. Temporary files
are created next to their destination, so publication remains atomic without
cross-directory renames (which Landlock ABI 1 forbids).
"""
import os
import tempfile

from triton.runtime.cache import FileCacheManager


class SameDirectoryCacheManager(FileCacheManager):
    def put(self, data, filename, binary=True):
        if not self.cache_dir:
            raise RuntimeError('Could not create or locate cache dir')
        if os.path.basename(filename) != filename:
            raise ValueError('Triton cache filenames must be simple basenames')
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        filepath = self._make_path(filename)
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.cache_dir)
        try:
            with os.fdopen(fd, 'wb' if binary else 'w') as stream:
                stream.write(data)
            os.replace(temporary, filepath)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return filepath
