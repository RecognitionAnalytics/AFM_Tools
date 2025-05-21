
import os
from array import array
from .FlammarionFile import FlammarionFile , FlammarionImageData
import numpy as np
import struct
from typing import Dict, Any

def loadMI(filename):
    size = os.path.getsize(filename)
    current = []

    baseTime = os.path.getctime(filename)
    distance = []
    times = []
    velocity = []
    distances = []
    data = array("B")
    isSpectroscopy = False
    bufferLabels = []
    bufferUnits = []
    bufferDirections = []
    bufferRange = []
    displayRange = []
    chunkLabels = []
    displayOffset = []
    parameters = {}
    with open(filename, "rb") as file1:
        # Reading form a file
        d = file1.readline()
        chunks = []
        while d != "data          BINARY\n" and d != "data          BINARY_32\n":
            d = file1.readline()

            d = d.decode("ascii")
            dd = d.strip().split(" ")

            if len(dd) > 1:
                ddClean = (
                    " ".join([x for x in dd[1:] if x != " "])
                    .strip()
                    .replace("FALSE", "False")
                    .replace("TRUE", "True")
                )

                try:
                    ddClean = eval(ddClean)
                except:
                    pass

                parameters[dd[0]] = ddClean

            if len(dd) > 1 and dd[-1] == "Spectroscopy":
                isSpectroscopy = True

            if dd[0] == "bufferLabel":
                bufferLabels.append(" ".join(dd[1:]).strip())
            if dd[0] == "bufferUnit":
                bufferUnits.append(" ".join(dd[1:]).strip())
            if dd[0] == "direction":
                bufferDirections.append(" ".join(dd[1:]).strip())
            if dd[0] == "bufferRange":
                bufferRange.append(float(dd[-1]))

            if dd[0] == ("DisplayRange"):
                displayRange.append(float(dd[-1]))
            if dd[0] == ("DisplayOffset"):
                displayOffset.append(float(dd[-1]))

            if dd[0] == ("bias"):
                bias = float(dd[-1])

            if d.startswith("chunk"):
                parts = d.split("\t")

                dist = float(parts[5])
                time0 = float(parts[2])
                time = float(parts[3])
                valueO = float(parts[4])
                label = parts[-1].strip()
                chunks.append(
                    {
                        "points": int(parts[1]),
                        "distperpoint": dist,
                        "timeperpoint": time,
                        "startValue": valueO,
                        "time0": time0,
                        "label": label,
                    }
                )

        loc = file1.tell()
        binarySize = size - loc
        data.fromfile(file1, binarySize)

        if not isSpectroscopy:
            cc = 0
            xPixels = parameters["xPixels"]
            yPixels = parameters["yPixels"]
            file = FlammarionFile()
            file.filename = filename
            file.datasetName = os.path.basename(filename).split(".")[0]
            file.metaData = parameters
            file.resolution = (xPixels, yPixels)
            file.physicalSize = (parameters["xLength"], parameters["yLength"])
            file.physicalSizeUnit = "m"

            for i in range(0, len(bufferLabels)):
                img = np.zeros((xPixels, yPixels))

                for j in range(0, xPixels):
                    for k in range(0, yPixels):
                        img[j, k] = struct.unpack("i", data[cc : cc + 4])[0]
                        cc += 4

                fImg = FlammarionImageData()
                fImg.data = img * bufferRange[i] / 2147483648.0
                if bufferUnits[i] == "um":
                    fImg.data = fImg.data * 1e-6
                    fImg.zunit = "m"
                else:
                    fImg.zunit = bufferUnits[i]
                fImg.label = bufferLabels[i]
                fImg.direction = bufferDirections[i]
                fImg.physicalSize = (parameters["xLength"], parameters["yLength"])
                fImg.physicalSizeUnit = "m"
                file.images.append(fImg)

            return file
        else:
            i = 0
            while True:
                cDist = 0

                for chunk in chunks:
                    chunkDist = []
                    chunkCurrent = []

                    if isSpectroscopy:
                        cDist = chunk["startValue"]

                    for j in range(0, chunk["points"]):
                        try:
                            f = struct.unpack("<f", data[i : i + 4])
                        except:
                            print(chunkLabels)
                            curveChunks = []
                            for k in range(len(distance)):
                                curveChunks.append(
                                    {
                                        "x": np.array(distance[k]),
                                        "y": np.array(current[k]),
                                        "time": np.array(times[k]),
                                        "velocity": velocity[k],
                                        "label": chunkLabels[k],
                                    }
                                )

                            curves = {
                                "xlabel": bufferLabels[0],
                                "xunit": bufferUnits[0],
                                "ylabel": bufferLabels[1],
                                "yunit": bufferUnits[1],
                                "data": curveChunks,
                                "time": np.array(times[k]),
                                "velocity": np.array(velocity[k]),
                            }
                            return {"curves": curves, "parameters": parameters}
                        chunkCurrent.append(f)
                        cDist += chunk["distperpoint"]
                        chunkDist.append(cDist)
                        i += 4

                    dist = np.array(chunkDist)
                    if len(dist) > 0:
                        chunkLabels.append(chunk["label"])
                        distance.append(dist)
                        current.append(np.array(chunkCurrent).ravel())
                        totalTime = len(chunkDist) * chunk["timeperpoint"]
                        time = chunk["time0"] + np.linspace(
                            0, totalTime, len(chunkDist)
                        )

                        times.append(time)
                        if len(dist) > 0:
                            distances.append((dist[0] - dist[-1]))
                            velocity.append((dist[0] - dist[-1]) / (time[0] - time[-1]))