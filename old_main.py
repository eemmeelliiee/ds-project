from data_preparation import PrepareData
from data_modeling_and_model_evalutation import ValidateFeatureImportance, DropUnimportantFeatures, ModelDataAndEvaluate
from data_exploration import ExploreData 

def main():
    PrepareData()
    ModelDataAndEvaluate()
    ValidateFeatureImportance()
    DropUnimportantFeatures()
    ExploreData()

if __name__ == "__main__":
    main()