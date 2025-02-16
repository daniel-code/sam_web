import {FileProps} from "./helpers/Interfaces";
import {useContext} from "react";
import AppContext from "./hooks/createContext";
import React from "react";
import Tool from "./Tool";

const UploadImage = ({handleFileChange}: FileProps) => {
    const {
        clicks: [, setClicks],
        image: [image],
    } = useContext(AppContext)!;

    return (
        <div className="flex items-center justify-center w-full h-full">
            <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFileChange(e)}
            />
        </div>
    );
}

export default UploadImage;
