# Alzheimer Project Overview

This repository hosts two distinct projects: `10x RNA-seq Gene Expression Data Project` and `Alzheimer Research Data Processing`. Each project leverages specific datasets to explore and analyze gene expression and medical imaging data related to neurological studies.

[Alzheimer Documentation](./doc.md)

## Projects

### 1. Alzheimer Research Data Processing

#### Description
This project focuses on processing and analyzing data specifically related to Alzheimer's disease research. It utilizes machine learning models to predict disease stages based on MRI brain image data and other biomarkers.

#### Data
- **Files:**
  - `test.parquet`
  - `train.parquet`
  - These datasets are used within the `Alzheimer_Research_Data_Processing.ipynb` notebook for developing models that predict Alzheimer's disease progression from MRI images.

#### Requirements
- Python environment with libraries such as `torch`, `numpy`, `pandas`, `matplotlib`, and `cv2` (OpenCV).

[Requirements](./requirements.txt)

### 2. 10x RNA-seq Gene Expression Data Project

#### Description
This project explores RNA-seq gene expression data from 4 million cells using 10x Genomics technology. The data are categorized based on 10X chemistry (10Xv2, 10Xv3, and 10XMulti) and the broad anatomical region from which the cells were sourced. The Jupyter Notebook included provides an overview of the data, file organization, and practical examples of how to combine data with metadata for analysis.

#### Data
- **File:** `ABC_Atlas_Class_01_IT-ET_Glut_cells_2024_10_23_16_42.csv`
  - Used for analysis in the `10x_snRNASeq.ipynb` notebook.
  - Contains single-nucleus RNA sequencing data, which helps in identifying cellular components affected in neurological disorders.

#### Requirements
- Internet connection
- Jupyter Notebook
- Python libraries: `pandas`, `numpy`, `matplotlib`, `anndata`

[Requirements](./requirements.txt)

## General Information

### File Structure

- `Data/`: Contains datasets and metadata.
  - `ABC_Atlas_Class_01_IT-ET_Glut_cells_2024_10_23_16_42.csv`: Cell data used in the `10x_snRNASeq.ipynb` notebook for single-nucleus RNA analysis.
  - `train.parquet` and `test.parquet`: Datasets used for training and testing models to predict Alzheimer's disease progression from MRI images, used in the `Alzheimer_Research_Data_Processing.ipynb` notebook.
- `Images/`: Directory containing images used in the doc.md documentation.
- `10x_snRNASeq.ipynb`: Notebook for RNA-seq data analysis.
- `Alzheimer_Research_Data_Processing.ipynb`: Notebook for processing MRI data and training artificial intelligence models.
- `LICENSING`: The document describing the terms of use for the data and source code under the MIT license.
- `README.md`: The main documentation of the project, providing an overview of the purpose, file structure, and usage instructions.
- `requirements.txt`: A file that lists the libraries needed to run the project's notebooks.
- `doc.md`: An additional Markdown document with detailed information about data analysis and project outcomes.

### Usage
Change the `download_base` variable to the local path where the data has been downloaded in your system. Use `AbcProjectCache` to interact with downloaded data.

### Licensing
This project is available under the MIT License. See the LICENSING file for more details.



