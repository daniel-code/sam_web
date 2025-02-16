import base64
from typing import Annotated, Any, List, Sequence, Union

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from segment_anything import SamPredictor, sam_model_registry
from starlette.middleware.cors import CORSMiddleware

"""
The `image_to_base64` and `base64_to_image` functions are referenced from the following link:
https://annacsmedeiros.medium.com/efficient-image-processing-in-python-a-straightforward-guide-to-base64-and-numpy-conversions-e9e3aac13312

"""


def image_to_base64(
    image,
    ext=".png",
    encode_params: Sequence[int] = (int(cv2.IMWRITE_PNG_COMPRESSION), 9),
) -> str:
    _, buffer = cv2.imencode(ext, image, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(base64_image: str) -> np.ndarray:
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_image)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


class _NpNdArrayTypePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_from_str(value: Union[str, np.ndarray]) -> np.ndarray:
            if isinstance(value, str):
                np_array = base64_to_image(value)
            else:
                np_array = value
            assert isinstance(np_array, np.ndarray), f"Expected an instance of `np.ndarray`, but got {type(np_array)}"
            return np_array

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(validate_from_str),
            python_schema=core_schema.no_info_plain_validator_function(validate_from_str),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: (image_to_base64(instance) if isinstance(instance, np.ndarray) else instance)
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(
            core_schema.union_schema(
                [
                    core_schema.list_schema(
                        core_schema.list_schema(
                            core_schema.list_schema(
                                core_schema.union_schema(
                                    [
                                        core_schema.int_schema(),
                                    ]
                                ),
                            ),
                        ),
                    ),
                    core_schema.str_schema(),
                ]
            )
        )


NdArray = Annotated[Union[str, np.ndarray], _NpNdArrayTypePydanticAnnotation]


class ImageRequest(BaseModel):
    image: NdArray


class EmbeddingResponse(BaseModel):
    embedding: List[List[List[List[float]]]]


sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

predictor = SamPredictor(sam)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/embedding", response_model=EmbeddingResponse)
async def get_embedding(image_request: ImageRequest):
    image = cv2.cvtColor(image_request.image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    return EmbeddingResponse(embedding=image_embedding.tolist())
