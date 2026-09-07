import onnxruntime as ort

DML_PROVIDER = "DmlExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

DEVICE_DML = "dml"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

PROVIDER_BY_DEVICE = {DEVICE_DML: DML_PROVIDER, DEVICE_CUDA: CUDA_PROVIDER,
                      DEVICE_CPU: CPU_PROVIDER}

_patched = False


def patch_rtmlib():
    global _patched
    if _patched:
        return
    from rtmlib.tools.base import RTMLIB_SETTINGS

    RTMLIB_SETTINGS["onnxruntime"][DEVICE_DML] = DML_PROVIDER
    RTMLIB_SETTINGS["onnxruntime"][DEVICE_CUDA] = CUDA_PROVIDER
    _patched = True


def available_providers():
    return list(ort.get_available_providers())


def has_directml():
    return DML_PROVIDER in available_providers()


def has_cuda():
    return CUDA_PROVIDER in available_providers()


def resolve_device(preferred=DEVICE_DML, fallback=DEVICE_CPU):
    provider = PROVIDER_BY_DEVICE.get(preferred)
    if provider is None:
        return fallback, f"unknown device '{preferred}', falling back"
    if provider in available_providers():
        return preferred, None
    if preferred == DEVICE_DML:
        return fallback, (
            "DirectML requested but DmlExecutionProvider is not available. "
            "Install onnxruntime-directml (not plain onnxruntime).")
    if preferred == DEVICE_CUDA:
        return fallback, (
            "CUDA requested but CUDAExecutionProvider is not available. "
            "Install onnxruntime-gpu built against the installed CUDA and "
            "cuDNN, and make sure plain onnxruntime is not shadowing it.")
    return fallback, f"{provider} is not available"


def session_providers(device):
    return [PROVIDER_BY_DEVICE.get(device, CPU_PROVIDER)]


def describe():
    return {
        "onnxruntime_version": ort.__version__,
        "available_providers": available_providers(),
        "directml": has_directml(),
    }
