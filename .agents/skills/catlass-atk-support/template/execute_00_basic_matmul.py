import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

############# Pre-defined data map #############

_OUT_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}

############# Helper functions #############

def out_dtype_to_torch(out_dtype: str) -> torch.dtype:
    if out_dtype not in _OUT_DTYPE_MAP:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")
    return _OUT_DTYPE_MAP[out_dtype]

def apply_npu_nz_format(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    *,
    format_a: bool = False,
    format_b: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NZ block format cast without physical transpose.

    torch_catlass kernels interpret ``transA``/``transB`` against the original
    storage shapes (see ``CatlassKernelWrapper::MatmulLike::GetKernelInfo`` in
    ``catlass/tests/optest/src/include/template/matmul.h``). Physical permute
    before the call would make the transpose flags apply twice and produce a
    wrong output shape.
    """
    import torch_npu

    if format_a:
        inputs = torch_npu.npu_format_cast(inputs, 29)
    if format_b:
        weights = torch_npu.npu_format_cast(weights, 29)
    return inputs, weights

############# ATK Test suite #############

@register("executeBasicMatmul")
class Api(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        inputs = input_data.kwargs["inputs"]
        weights = input_data.kwargs["weights"]

        out_dtype = input_data.kwargs.get("out_dtype", "float16")
        trans_a = input_data.kwargs.get("transA", False)
        trans_b = input_data.kwargs.get("transB", False)
        formatA = input_data.kwargs.get("formatA", False)
        formatB = input_data.kwargs.get("formatB", False)
        if self.device == "cpu":
            a, b = inputs.cpu(), weights.cpu()
            if trans_a:
                a = a.permute(1, 0)
            if trans_b:
                b = b.permute(1, 0)
            golden = torch.matmul(a, b)
            return golden.to(out_dtype_to_torch(out_dtype))
        
        if self.device == "gpu":
            a, b = inputs.to(f"cuda:{self.device_id}"), weights.to(f"cuda:{self.device_id}")
            if trans_a:
                a = a.permute(1, 0)
            if trans_b:
                b = b.permute(1, 0)
            golden = torch.matmul(a, b)
            return golden.to(out_dtype_to_torch(out_dtype))

        if self.device == "npu":
            import torch_catlass
            inputs, weights = apply_npu_nz_format(inputs, weights, format_a=formatA, format_b=formatB)
            torch.npu.synchronize()
            result = torch_catlass.basic_matmul(inputs, weights, out_dtype, trans_a, trans_b, formatA, formatB)
            torch.npu.synchronize()
            return result
