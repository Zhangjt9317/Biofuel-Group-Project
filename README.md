# Biofuel-Group-Project

### Predicting Thermo-Physical Properties According to Structure

#### The main purpose of this project is to predict some properties (flashing point and cetane number) according to the structure of biofuels.

#### 1. Use Cases
* Use Case 1: generate SMILE structure according to the input information (CAS number...)
* Use Case 2: generate the functional group it has according to the structure
* Use Case 3: classify the substance according to the structure (Jingtian, Chenggang)
* Use Case 4: predict the properties according to the structure (Cheng, Renlong)

#### 2. Components
* Component 1: 
  * a. Name: GUI
  * b. What it does: Guide user to use the software
  * c. Inputs: Identification information
  * d. Outputs: Results
  * e. How use other components: Classifier, Prediction

* Component 2: 
  * a. Name: Structure generator
  * b. What it does: Generate SMILES
  * c. Inputs: CAS number
  * d. Outputs: SMILES

* Component 3: 
  * a. Name: Descriptor generator
  * b. What it does: Generate functional groups
  * c. Inputs: Structure generator
  * d. Outputs: functional groups
  * e. How use other components: Structure generator

* Component 4: 
  * a. Name: Classifier
  * b. What it does: Classify families of biofuels
  * c. Inputs: Descriptors generator
  * d. Outputs: Families
  * e. How use other components: Descriptor generator, Machine Learning

* Component 5: 
  * a. Name: Prediction
  * b. What it does: Predict the properties of biofuels
  * c. Inputs: Descriptors generator
  * d. Outputs: Properties
  * e. How use other components: Descriptor generator, Machine Learning

* Component 6: 
  * a. Name: Machine Learning
  * b. What it does: Use the existing data to generate and train the model
  * c. Inputs: Database
  * d. Outputs: Model with high similarity
  * e. How use other components: Database

* Component 7: 
  * a. Name: Database
  * b. What it does: Compile data sets input into a data library
  * c. Inputs: Data files (.csv files or .dat files)
  * d. Outputs: a sorted, organized data library
  
#### 3. Diagram
