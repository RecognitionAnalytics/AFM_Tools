#pyFlammarion
============

**Advanced AFM Image Processing and Particle Analysis Toolkit**

`pyFlammarion` is a scientific Python toolkit for working with Atomic Force Microscopy (AFM) images. It provides a complete workflow for flattening, masking, segmenting, and analyzing surface data, with a focus on automation and precision.

> **Why "Flammarion"?**\
> The name is inspired by the famous 19th-century *Flammarion engraving*, which depicts a seeker peering beyond the flat earth into the hidden structure of the cosmos. Likewise, `pyFlammarion` aims to "peer beyond the surface" of AFM data---removing distortions, revealing hidden features, and enabling deeper insight into nanoscale structures.

* * * * *

🧰 Features
-----------

### 📁 AFM File Loaders

-   Will have support for multiple AFM file formats. (only support Agilent/Keysight/Molecular Image .mi files now)

-   Built-in normalization and unit conversion.

-   Easy access to both topography and metadata.

### 🧼 Automated Flattening

-   Multiple flattening techniques to correct image tilt, curvature, and scan-line noise:

    -   Polynomial plane subtraction

    -   Row/column leveling

    -   Adaptive line defect removal

-   Designed to preserve features while removing acquisition artifacts.

### 🩹 Defect Masking

-   Automatic masking of:

    -   Scan-line defects

    -   Saturated or clipped regions

    -   Edge artifacts

-   Manual override tools available for inspection and correction.

### ✂️ Segmentation Tools

-   **Edge-based segmentation**: Detects particle or domain boundaries using image gradients.

-   **Height-based segmentation**: Isolates features by elevation with customizable thresholds.

-   Combined workflows for robust segmentation in noisy datasets.

### ⚛️ Particle Analysis

-   Label, count, and characterize surface features.

-   Extract:

    -   Particle size distributions

    -   Area, perimeter, and eccentricity

    -   Spatial statistics (e.g., density, nearest-neighbor distances)

* * * * *

📦 Installation
---------------

bash

CopyEdit

`pip install git+https://github.com/RecognitionAnalytics/pyFlammarion.git`

Or clone the repo:

bash

CopyEdit

`git clone https://github.com/RecognitionAnalytics/pyFlammarion.git
cd pyFlammarion
`

* * * * *
 

📚 Documentation
----------------

Comprehensive documentation and examples are coming soon.

In the meantime, see the [`tutorials/`](https://github.com/RecognitionAnalytics/pyFlammarion/blob/main/Tutorials.ipynb) folder for sample workflows.

* * * * *

🤝 Contributing
---------------

We welcome contributions! Please submit a pull request or open an issue to suggest features or report bugs.

* * * * *

🧪 License
----------

MIT License --- free to use, modify, and distribute.

* * * * *

🌌 Acknowledgments
------------------

Thanks to the researchers, engineers, and microscopy experts whose insight shaped this toolkit.\
Special homage to Camille Flammarion, for reminding us that what seems flat is often just the surface.