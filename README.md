###  **Credit Risk Management Model**

###  Author: Umidjon Sattorov, Machine Learning Engineer, successful graduate of the SkillBox platform, and student at Mohirdev platform

**Overview**

This project is part of the final assignment from the SkillBox platform, focusing on developing a machine learning model for credit risk management. Through deep analysis of the provided dataset, various conclusions were drawn, leading to the creation of an effective model using CatBoost. Despite the dataset's significant imbalance, the model achieved an impressive ROC AUC score of over 0.75. Techniques such as weight assignment were employed to ensure the model remained unbiased.

**Features**

High Accuracy: Achieved a ROC AUC score of more than 0.75, showcasing strong model performance.
Imbalanced Dataset Handling: Utilized weight assignment techniques to address the dataset's imbalance and reduce model bias.
Comprehensive Analysis: Performed in-depth data analysis to derive valuable insights and inform model development.
Robust Validation: Included thorough validation processes to ensure the model's reliability and effectiveness.

**Technologies Used**

Python: Core programming language for data processing and model development.
pandas: Data manipulation and analysis.
scikit-learn: Machine learning library for model training and evaluation.
CatBoost: Gradient boosting library for handling categorical features and imbalanced datasets.
FastAPI: Framework for creating APIs to deploy the model.

**File Structure and Descriptions**

main_final - Russian.ipynb: Jupyter notebook containing the final model development and evaluation in Russian.
main_final.ipynb: Jupyter notebook with the final model development and evaluation in English.
Model_validation_pkl.ipynb: Jupyter notebook for validating the pickle model preserved after running the pipeline code.
pipeline: Contains all files related to the preprocessing steps and final model pipeline, ensuring streamlined and reproducible workflows.
FastAPI server: Implementation of a FastAPI server to deploy and test the model, ensuring seamless integration and outstanding results.

**Results**

The model's high ROC AUC score of over 0.75 underscores its ability to accurately predict credit risk, providing a valuable tool for financial institutions to assess and manage risk effectively. The project demonstrates a comprehensive approach to handling imbalanced datasets and showcases the use of advanced machine learning techniques to achieve reliable and interpretable results.

From the graphs below, you can see how my catboost algorithms performed in the prediction of credit risk management however there are many issues with dataset.(Dataset were so big, that I coudn't work in my local computer because of lack of computational resources, thus I purchased some computational units from Google colab service) : 

![image](https://github.com/user-attachments/assets/3ad690a2-f118-48bd-b5a7-cc6ca036ff37)

![image](https://github.com/user-attachments/assets/5fdfdf4b-b7f2-4603-95c1-4577724b1a57)

![image](https://github.com/user-attachments/assets/e3e803c9-1136-46fb-9798-c7f4730d5ca2)

Pipeline structure : 

![image](https://github.com/user-attachments/assets/8c743a4e-9e7f-4b7e-9f43-9f8b68c852ad)

**Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

**Info**

This project was assigned by SkillBox and evaluated by highly qualified experts. It was successfully accepted by the mentor, and I am certified for the completion of the "Machine Learning engineering Junior" course. Here is the certification from SkillBox:

![certifical_completion](https://github.com/user-attachments/assets/9bb28498-bde7-44bc-a1e5-dc3d0188c505)

More information about the project and my approach can be found in the presentation included in the presentations folder.
