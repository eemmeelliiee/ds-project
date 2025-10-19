# ds-project-2
Access the dataset here [https://www.kaggle.com/datasets/rishabchitloor/indian-water-quality-data-2021-2023]

Expected structure:
```
├── ds-project-2
│   ├── data
│   │   ├── original_dataset.csv
```
(Note the explected name of the .csv-file)

## Predicting Safe Water Sources in India using a Decision Tree
This project aims to predict the safety level of water sources across India using a supervised classification model; a Decision Tree Classifier trained on publicly available environmental data.
By leveraging low-cost and easily measurable parameters such as temperature (min/max), pH (min/max), dissolved oxygen (min/max), type of water body, and state name, the model classifies each sample into one of three categories defined by the Central Pollution Control Board (CPCB):

- Class A – Safe to drink
- Class C – Safe to drink after conventional treatment
- Class Other – Unsafe to drink

The project demonstrates that accessible environmental indicators can reliably predict water quality (F1-score ≥ 0.84, balanced accuracy ≥ 0.88), offering a cost-effective tool for environmental monitoring and public health decision-making.

Main components:
```pipeline_and_model.py```: Preprocesses raw data, encodes categorical features, imputes missing values, and creates the target variable (water_quality). Also trains, evaluates, and explains the Decision Tree model using cross-validation, SHAP analysis, and calibration metrics.
```explore_data.py```: Generates visualizations to understand data distribution and feature relationships.

<!--- ## Instructions:
Fork the repo. The original dataset is included in folder ```data/``` and thus does not need to be downloaded.

Running the main script calls the following methods:
1. ```PrepareData()``` from ```data_preparation.py``` which cleans the original dataset and outputs ```cleaned_dataset.csv``` (which is used in the remaining modules).
2. ```ModelDataAndEvaluate()``` from ```data_modeling–and–model_evaluation.py```, which models a decision tree and evaluates the resuls.
3. ```ExploreData()``` from ```data_evaluation.py```, which saves 5 insightful visualizations to ```figures/data-exploration/``` folder.



 -->
