# bluebottle-xai
Bluebottle migration analysis using explainable artificial intelligence. The project is trying to predict and determine factors that contributes to the presence of Bluebottles(Physalia Physalis) in Australia considering data from Randwick Council. It is a classification problem(Absence and Presence) of Bluebottle and different classification algorithm will be considered. In this study, we employ machine learning to explore the environmental factors influencing the arrival of bluebottle marine organisms on Australian beaches. To tackle challenges associated with this data such as class imbalance, overlap and unreliable absence data, we utilize data augmentation techniques, including SMOTE, random samoling and a "no-negative class" approach.

# Folder Structure and Contents
The repository includes three main folders that organise the projects's codebase: **Bluebottle Data**, **Model Training** and **Plots**. 

1. **Bluebottle Data**
   
  This folder contains the datasets used in the study, including:
  * **Raw data:** The Original, unprocessed dataset
  * **Preprocessed data:** The cleaned and prepared dataset.
  * **CT-GAN generated data:** The synthetic data created using the CT-GAN model.

2. **Model_Training**
   
  This folder includes the python scripts for the data analysis, Discretization and Modelling:
  * `Exploratory_analysis.py`: Performs data analysis witht the code modularized into functions for clarity.
    
  * `CT-GAN_SVM.py`: Generates synthetic negative classes using CT-GAN and includes a function for One-Class SVM modeling.
    
  * `discretizer.py`: Converts continuous variables into discrete categories to identify subfeatures contributing to bluebottle presence.
    
  * `Model_bluebottledata.py`: Implements the entire training pipeline using various augmentatio techniques and a baseline without augmentation.
    
  * `packages.py`: Lists all the packages utilized in the study.

3. **Plots**
   
  This folder contains all the plots generated fromt the data analysis and modelling. 
