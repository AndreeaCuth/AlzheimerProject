
# Genetic Mechanisms and Risk Factors in Alzheimer's Disease: Monogenic Pathways, Oligogenic Inheritance, and Genome-Wide Associations

## Authors
- **Authors**: Szoke Mark-Andor, Andreea-Maria Cuth

## Overview of the Proposed Paradigm
- Alzheimer's Disease (AD) is described as the most common form of dementia, having a profound impact on society. The research focuses on understanding the genetic complexity contributing to the disease's development.
- The proposed paradigm centers on exploring monogenic genetic pathways and identifying oligogenic factors, along with leveraging genome-wide association data to uncover novel therapeutic targets.

## Specific Architectural Description
- The document explores the implications of mutations in the **APP**, **PSEN1**, and **PSEN2** genes, which can cause monogenic forms of Alzheimer's Disease, with autosomal dominant inheritance significantly influencing the disease's probability.
- The analysis includes contributions from oligogenic variations, which combine the effects of multiple genes to influence the disease risk, as well as methods for correlating these genes with clinical phenotypes.

## Project Configuration
- The formulated problem focuses on identifying and thoroughly analyzing genetic variants contributing to Alzheimer's susceptibility, emphasizing the distinction between monogenic and oligogenic contributions.
- The data used includes genetic datasets from large cohorts of patients, as well as high-resolution sequencing data available through international projects.
- The technologies involved include next-generation sequencing (NGS) analyses, advanced bioinformatics methods for genetic data analysis, and biomedical database management systems.

## Code, Important Sections

1. **Alzheimer_Research_Data_Processing**

### Loading Training and Test Datasets
```python
df_train = pd.read_parquet('Data/train.parquet')
df_test = pd.read_parquet('Data/test.parquet')
```
This code loads the training and test datasets from the train.parquet and test.parquet files. 
These datasets are used to train and evaluate the machine learning model.

### Defining Transformations for Image Preprocessing
```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```
This code configures transformations applied to images, including conversion to PIL format, resizing, tensorization, and normalization. 
These preprocessing steps are essential for standardizing input data before feeding it into the model.

### Defining a Simple CNN Model for MRI Image Classification
```python
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 64 * 64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)
        x = self.fc(x)
        return x
```       
The model defined here is a simple convolutional neural network (CNN) that includes convolutional layers, pooling layers, and a fully connected layer, designed for classifying MRI images based on detected features.

### Training the Model
```python
for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```       
This code demonstrates the model's training loop, using images and labels from the dataset and displaying the loss for each epoch to evaluate the model's learning progress.

2. **10x_snRNASeq**

### Initial Loading and Processing of Data
```python
data = sc.read_10x_h5('data_path')
sc.pp.filter_cells(data, min_genes=200)
```
This section handles the import of snRNASeq data, focusing on data structures and initial exploration. 
The code reads the data from a specific location and displays the initial rows to provide an overview of the dataset structure.

### Data Preprocessing
```python
sc.pp.normalize_total(data, target_sum=1e4)
sc.pp.log1p(data)
```
The data is preprocessed to filter cells based on quality metrics, such as the number of detected genes. 
This step ensures high data quality and robust analyses.

### Normalization and Feature Selection
```python
sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)
data = data[:, data.var.highly_variable]
```       
Normalization techniques are applied to the dataset to minimize differences in gene expression levels caused by varying sequencing depths. 
Additionally, key features (genes) are selected based on variability, helping to identify the most informative features for further analysis.

### Dimensionality Reduction
```python
sc.tl.pca(data, svd_solver='arpack')
sc.pp.neighbors(data, n_neighbors=10, n_pcs=40)
sc.tl.umap(data)
```       
Techniques such as PCA (Principal Component Analysis) and UMAP (Uniform Manifold Approximation and Projection) are used to reduce the dataset's dimensionality. 
This simplifies data complexity, facilitating visualization and identifying patterns or clusters.

### Cluster Analysis
```python
sc.tl.louvain(data)
sc.pl.umap(data, color='louvain')
```       
The dataset is grouped to identify clusters of similar cells, which may indicate different cell types or states. 
This is a critical step in snRNASeq data analysis, providing insights into the cellular composition of samples.

### Differential Expression Analysis
```python
sc.tl.rank_genes_groups(data, 'louvain', method='t-test')
sc.pl.rank_genes_groups(data, n_genes=25, sharey=False)
```       
After clustering, differential expression analysis identifies genes significantly up- or down-regulated between clusters. 
This step characterizes biological differences between the identified cell types or conditions.

### Annotation and Interpretation
```python
data.obs['cell_type'] = data.obs['louvain'].map(cell_type_dict)
sc.pl.umap(data, color='cell_type')
```       
Finally, clusters are annotated based on known genetic markers, and results are interpreted to draw biological conclusions. 
This step links computational analysis to biological implications.

## Application Implementation
   - Screenshots from the software tools used for analyses are provided, demonstrating data processing steps and obtained results.

1. **Alzheimer_Research_Data_Processing**

![](Images/p1.PNG)
![](Images/p2.PNG)
![](Images/p34.PNG)
![](Images/p4.PNG)



2. **10x_snRNASeq**



![](Images/s1.PNG)
![](Images/s2.PNG)
![](Images/s3.PNG)
![](Images/s4.PNG)
![](Images/s5.PNG)
![](Images/s6.PNG)
![](Images/s7.PNG)


