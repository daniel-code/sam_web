import click
from onnxruntime.quantization import QuantType, quantize_dynamic


@click.command()
@click.argument("onnx_model_path", type=click.Path(exists=True))
@click.argument("onnx_model_quantized_path", type=click.Path())
def main(onnx_model_path, onnx_model_quantized_path):
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=onnx_model_quantized_path,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    main()
