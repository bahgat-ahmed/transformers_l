"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.util
import json
import os
import shutil
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any

from packaging import version

from transformers.utils.versions import importlib_metadata

from . import logging_t

logger = logging_t.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

_torch_version = "N/A"
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_version = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


_tf_version = "N/A"
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
            candidates = (
                "tensorflow",
                "tensorflow-cpu",
                "tensorflow-gpu",
                "tf-nightly",
                "tf-nightly-cpu",
                "tf-nightly-gpu",
                "intel-tensorflow",
                "intel-tensorflow-avx512",
                "tensorflow-rocm",
                "tensorflow-macos",
                "tensorflow-aarch64",
            )
            _tf_version = None
            # For the metadata, we have to look for both tensorflow and tensorflow-cpu
            for pkg in candidates:
                try:
                    _tf_version = importlib_metadata.version(pkg)
                    break
                except importlib_metadata.PackageNotFoundError:
                    pass
            _tf_available = _tf_version is not None
        if _tf_available:
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(
                    f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum."
                )
                _tf_available = False
            else:
                logger.info(f"TensorFlow version {_tf_version} available.")
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")
        _tf_available = False


if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available = (
        importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
    )
    if _flax_available:
        try:
            _jax_version = importlib_metadata.version("jax")
            _flax_version = importlib_metadata.version("flax")
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _flax_available = False
else:
    _flax_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
try:
    # Check we're not importing a "datasets" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    _ = importlib_metadata.version("datasets")
    _datasets_metadata = importlib_metadata.metadata("datasets")
    if _datasets_metadata.get("author", "") != "HuggingFace Inc.":
        _datasets_available = False
except importlib_metadata.PackageNotFoundError:
    _datasets_available = False


_detectron2_available = importlib.util.find_spec("detectorn2") is not None
try:
    _detectron2_version = importlib_metadata.version("detectron2")
    logger.debug(f"Successfully imported detectron2 version {_detectron2_version}")
except importlib_metadata.PackageNotFoundError:
    _detectron2_available = False


_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib_metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib_metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib_metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib_metadata.PackageNotFoundError:
        _faiss_available = False


_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False

coloredlogs = importlib.util.find_spec("coloredlogs") is not None
try:
    _coloredlogs_available = importlib_metadata.version("coloredlogs")
    logger.debug(f"Successfully imported sympy version {_coloredlogs_available}")
except importlib_metadata.PackageNotFoundError:
    _coloredlogs_available = False


sympy_available = importlib.util.find_spec("sympy") is not None
try:
    _sympy_available = importlib_metadata.version("sympy")
    logger.debug(f"Successfully imported sympy version {_sympy_available}")
except importlib_metadata.PackageNotFoundError:
    _sympy_available = False


_tf2onnx_available = importlib.util.find_spec("tf2onnx") is not None
try:
    _tf2onnx_version = importlib_metadata.version("tf2onnx")
    logger.debug(f"Successfully imported tf2onnx version {_tf2onnx_version}")
except importlib_metadata.PackageNotFoundError:
    _tf2onnx_available = False


_onnx_available = importlib.util.find_spec("onnxruntime") is not None
try:
    _onxx_version = importlib_metadata.version("onnx")
    logger.debug(f"Successfully imported onnx version {_onxx_version}")
except importlib_metadata.PackageNotFoundError:
    _onnx_available = False


_pytorch_quantization_available = importlib.util.find_spec("pytorch_quantization") is not None
try:
    _pytorch_quantization_version = importlib_metadata.version("pytorch_quantization")
    logger.debug(
        f"Successfully imported pytorch-quantization version {_pytorch_quantization_version}"
    )
except importlib_metadata.PackageNotFoundError:
    _pytorch_quantization_available = False


_soundfile_available = importlib.util.find_spec("soundfile") is not None
try:
    _soundfile_version = importlib_metadata.version("soundfile")
    logger.debug(f"Successfully imported soundfile version {_soundfile_version}")
except importlib_metadata.PackageNotFoundError:
    _soundfile_available = False


_tensorflow_probability_available = importlib.util.find_spec("tensorflow_probability") is not None
try:
    _tensorflow_probability_version = importlib_metadata.version("tensorflow_probability")
    logger.debug(
        f"Successfully imported tensorflow-probability version {_tensorflow_probability_version}"
    )
except importlib_metadata.PackageNotFoundError:
    _tensorflow_probability_available = False


_timm_available = importlib.util.find_spec("timm") is not None
try:
    _timm_version = importlib_metadata.version("timm")
    logger.debug(f"Successfully imported timm version {_timm_version}")
except importlib_metadata.PackageNotFoundError:
    _timm_available = False


_natten_available = importlib.util.find_spec("natten") is not None
try:
    _natten_version = importlib_metadata.version("natten")
    logger.debug(f"Successfully imported natten version {_natten_version}")
except importlib_metadata.PackageNotFoundError:
    _natten_available = False


_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
try:
    _torchaudio_version = importlib_metadata.version("torchaudio")
    logger.debug(f"Successfully imported torchaudio version {_torchaudio_version}")
except importlib_metadata.PackageNotFoundError:
    _torchaudio_available = False


_phonemizer_available = importlib.util.find_spec("phonemizer") is not None
try:
    _phonemizer_version = importlib_metadata.version("phonemizer")
    logger.debug(f"Successfully imported phonemizer version {_phonemizer_version}")
except importlib_metadata.PackageNotFoundError:
    _phonemizer_available = False


_pyctcdecode_available = importlib.util.find_spec("pyctcdecode") is not None
try:
    _pyctcdecode_version = importlib_metadata.version("pyctcdecode")
    logger.debug(f"Successfully imported pyctcdecode version {_pyctcdecode_version}")
except importlib_metadata.PackageNotFoundError:
    _pyctcdecode_available = False


_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    _librosa_version = importlib_metadata.version("librosa")
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    _librosa_available = False

ccl_version = "N/A"
_is_ccl_available = (
    importlib.util.find_spec("torch_ccl") is not None
    or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
)
try:
    ccl_version = importlib_metadata.version("oneccl_bind_pt")
    logger.debug(f"Successfully imported oneccl_bind_pt version {ccl_version}")
except importlib_metadata.PackageNotFoundError:
    _is_ccl_available = False

_decord_availale = importlib.util.find_spec("decord") is not None
try:
    _decord_version = importlib_metadata.version("decord")
    logger.debug(f"Successfully imported decord version {_decord_version}")
except importlib_metadata.PackageNotFoundError:
    _decord_availale = False
