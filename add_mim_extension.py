import os
import os.path as osp
import shutil
import sys
import warnings

from setuptools.command.build import build


class MIMBuild(build):
    """Custom build command."""
    def initialize_options(self) -> None:
        self._pre_initialize_options()
        build.initialize_options(self)

    def _pre_initialize_options(self) -> None:
        """Add extra files that are required to support MIM into the package.

        These files will be added by creating a symlink to the originals if the
        package is installed in `editable` mode (e.g. pip install -e .), or by
        copying from the originals otherwise.
        """
        print("Adding MIM extension...", sys.argv)

        # parse installment mode
        if "develop" in sys.argv:
            # installed by `pip install -e .`
            mode = "symlink"
        elif "sdist" in sys.argv or "bdist_wheel" in sys.argv:
            # installed by `pip install .`
            # or create source distribution by `python setup.py sdist`
            mode = "copy"
        else:
            return

        filenames = ["tools", "configs"]
        repo_path = osp.dirname(__file__)
        mim_path = osp.join(repo_path, "diffengine", ".mim")
        os.makedirs(mim_path, exist_ok=True)

        for filename in filenames:
            if osp.exists(filename):
                src_path = osp.join(repo_path, filename)
                tar_path = osp.join(mim_path, filename)

                if osp.isfile(tar_path) or osp.islink(tar_path):
                    os.remove(tar_path)
                elif osp.isdir(tar_path):
                    shutil.rmtree(tar_path)

                if mode == "symlink":
                    src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                    try:
                        os.symlink(src_relpath, tar_path)
                    except OSError:
                        # Creating a symbolic link on windows may raise an
                        # `OSError: [WinError 1314]` due to privilege. If
                        # the error happens, the src file will be copied
                        mode = "copy"
                        warnings.warn(
                            f"Failed to create a symbolic link for {src_relpath}, "
                            f"and it will be copied to {tar_path}")
                    else:
                        continue

                if mode == "copy":
                    if osp.isfile(src_path):
                        shutil.copyfile(src_path, tar_path)
                    elif osp.isdir(src_path):
                        shutil.copytree(src_path, tar_path)
                    else:
                        warnings.warn(f"Cannot copy file {src_path}.")
                else:
                    msg = f"Invalid mode {mode}"
                    raise ValueError(msg)
