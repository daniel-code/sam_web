# Segmentation Anything Model (SAM) Web Demo

This project is a web demo for the Segmentation Anything Model (SAM)] based
on [Segmentation Anything Model (SAM) repository](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf/demo).

<img src="https://github.com/daniel-code/sam_web/raw/main/assets/demo.gif" width="500"/>

## Usage

### Pre-requisites

1. Download the SAM model checkpoint from
   the [SAM repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
2. Convert the checkpoint to ONNX format following the instructions in the
   [SAM repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#onnx-export)
3. Quantize the ONNX model to INT8 format.

   ```shell
   python onnx_model_quantized.py sam_vit_b_01ec64.onnx sam_vit_b_01ec64_QUInt8.onnx
   ```

4. Copy the quantized ONNX model to the `demo/model` directory.

Here is the final directory structure:

```
.
├── demo       # Web demo UI
│   ├── model  # Quantized ONNX model
│   │   └── sam_vit_b_01ec64_QUInt8.onnx
│   └── src    # Web demo source code
├── sam_vit_b_01ec64.onnx     # SAM onnx model
├── sam_vit_b_01ec64.pth      # SAM model checkpoint
├── main.py                   # Web demo server
└── onnx_model_quantized.py   # Quantize the ONNX model
```

**Note**: Based on the model checkpoint you have, you may have different model names.

### Run the Web Demo

**Backend**

1. Install the required packages.

   ```shell
   pip install -r requirements.txt
   ```
2. Run the web demo server.

   ```shell
    python -m uvicorn main:app --reload 
    ```

**Frontend**

see [demo/README.md](./demo/README.md)

## License

The python code in this project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The web demo code is licensed under the Apache-2.0 license - see
the [segment-anything LICENSE](https://github.com/facebookresearch/segment-anything?tab=Apache-2.0-1-ov-file#readme)
file for details.
