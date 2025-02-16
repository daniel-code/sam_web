// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import {InferenceSession, Tensor} from "onnxruntime-web";
import React, {useContext, useEffect, useState} from "react";
import "./assets/scss/App.scss";
import {handleImageScale} from "./components/helpers/scaleHelper";
import {modelScaleProps} from "./components/helpers/Interfaces";
import {onnxMaskToImage} from "./components/helpers/maskUtils";
import {modelData} from "./components/helpers/onnxModelAPI";
import Stage from "./components/Stage";
import AppContext from "./components/hooks/createContext";

const ort = require("onnxruntime-web");
/* @ts-ignore */
import npyjs from "npyjs";


// Define image, embedding and model paths
const MODEL_DIR = "/model/sam_vit_b_01ec64_QUInt8.onnx";

const App = () => {
    const {
        clicks: [clicks],
        image: [, setImage],
        maskImg: [, setMaskImg],
    } = useContext(AppContext)!;
    const [model, setModel] = useState<InferenceSession | null>(null); // ONNX model
    const [tensor, setTensor] = useState<Tensor | null>(null); // Image embedding tensor

    // The ONNX model expects the input to be rescaled to 1024.
    // The modelScale state variable keeps track of the scale values.
    const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);

    // Initialize the ONNX model. load the image, and load the SAM
    // pre-computed image embedding
    useEffect(() => {
        // Initialize the ONNX model
        const initModel = async () => {
            try {
                if (MODEL_DIR === undefined) return;
                const URL: string = MODEL_DIR;
                const model = await InferenceSession.create(URL);
                setModel(model);
            } catch (e) {
                console.log(e);
            }
        };
        initModel();

    }, []);

    const getEmbedding = async (imageData: string) => {
        try {
            // Reinitialize the tensor state variable
            setTensor(null);
            console.log("fetching embedding")
            const response = await fetch("http://localhost:8000/api/embedding", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({image: imageData}),
            });
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            // Flatten the nested array
            const flattenArray = (arr: any[]): number[] => {
                return arr.reduce((acc, val) => Array.isArray(val) ? acc.concat(flattenArray(val)) : acc.concat(val), []);
            };

            const flattenedEmbedding = flattenArray(data.embedding);
            const embeddingArray = new Float32Array(flattenedEmbedding);

            if (embeddingArray.length !== 1 * 256 * 64 * 64) {
                throw new Error(`Expected embedding length of ${1 * 256 * 64 * 64}, but got ${embeddingArray.length}`);
            }

            const reshapedEmbedding = new ort.Tensor("float32", embeddingArray, [1, 256, 64, 64]);

            setTensor(reshapedEmbedding);
            console.log("embedding fetched")
        } catch (error) {
            console.error("There was a problem with the fetch operation:", error);
        }
    };

    const getImageDataInBase64 = (image: HTMLImageElement): string => {
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext("2d");
        ctx?.drawImage(image, 0, 0);
        return canvas.toDataURL("image/png").split(",")[1];
    };

    const loadImage = async (url: URL | File) => {
        try {
            const img = new Image();
            if (url instanceof File) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    if (typeof e.target?.result === "string") {
                        img.src = e.target.result;
                    }
                };
                reader.readAsDataURL(url);
            } else {
                img.src = url.href;
            }
            img.onload = () => {
                const {height, width, samScale} = handleImageScale(img);
                setModelScale({
                    height: height,  // original image height
                    width: width,  // original image width
                    samScale: samScale, // scaling factor for image which has been resized to longest side 1024
                });
                img.width = width;
                img.height = height;
                const base64Data = getImageDataInBase64(img);
                getEmbedding(base64Data);
                setImage(img);
            };
        } catch (error) {
            console.log(error);
        }
    };


    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            loadImage(file);
        }
    };

    // Decode a Numpy file into a tensor.
    const loadNpyTensor = async (tensorFile: string, dType: string) => {
        let npLoader = new npyjs();
        const npArray = await npLoader.load(tensorFile);
        const tensor = new ort.Tensor(dType, npArray.data, npArray.shape);
        return tensor;
    };

    // Run the ONNX model every time clicks has changed
    useEffect(() => {
        runONNX();
    }, [clicks]);

    const runONNX = async () => {
        try {
            if (
                model === null ||
                clicks === null ||
                tensor === null ||
                modelScale === null
            )
                return;
            else {
                // Prepare the model input in the correct format for SAM.
                // The modelData function is from onnxModelAPI.tsx.
                const feeds = modelData({
                    clicks,
                    tensor,
                    modelScale,
                });
                if (feeds === undefined) return;
                // Run the SAM ONNX model with the feeds returned from modelData()
                const results = await model.run(feeds);
                const output = results[model.outputNames[0]];
                // The predicted mask returned from the ONNX model is an array which is
                // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
                setMaskImg(onnxMaskToImage(output.data, output.dims[2], output.dims[3]));
            }
        } catch (e) {
            console.log(e);
        }
    };

    return <Stage handleFileChange={handleFileChange}/>;
};

export default App;
