import os
import git
import shutil

ASCEND_HOME_PATH = os.environ["ASCEND_HOME_PATH"]

CATLASS_REPO_URL = "https://gitee.com/ascend/catlass.git"

CATLASS_TEST_PATH = os.path.dirname(__file__)
CATLASS_TEST_TMP_PATH = os.path.join("/tmp", "catlass_test")
CATLASS_TEST_KERNEL_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "kernel")
CATLASS_TEST_INCLUDE_PATH = os.path.join(CATLASS_TEST_PATH, "csrc", "include")
CATLASS_TEST_KERNEL_EXAMPLES_PATH = os.path.join(CATLASS_TEST_PATH, "csrc", "examples")

CATLASS_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "catlass")
CATLASS_INCLUDE_PATH = os.path.join(CATLASS_PATH, "include")


def set_catlass_path(catlass_path: str):
    global CATLASS_PATH
    global CATLASS_INCLUDE_PATH
    CATLASS_PATH = catlass_path
    CATLASS_INCLUDE_PATH = os.path.join(CATLASS_PATH, "include")


def set_catlass_version(catlass_version: str, reset: bool = False):
    if reset:
        shutil.rmtree(CATLASS_PATH)
    if not os.path.exists(CATLASS_PATH):
        git.Repo.clone_from(CATLASS_REPO_URL, to_path=CATLASS_PATH)
    repo = git.Repo(CATLASS_PATH)
    repo.git.checkout()


from catlass_test.interface.function import *
