This repository provides the dataset, source code, and trained models associated with the study “Domain-Specific Fine-Tuning and Prompt-Based Learning: A Comparative Study for Developing Natural Language-Based BIM Information Retrieval Systems.” The resources are released to support reproducibility and further research on natural language interfaces for Building Information Modeling (BIM).

Repository Contents
Models and Checkpoints
best_state_dict_model.pt
Trained model checkpoint containing the best-performing parameters of the domain-specific fine-tuned TAPAS model.

Evaluation Notebooks

evaluate_bimqa_tapas.ipynb
Jupyter notebook for evaluating the fine-tuned TAPAS model on the BIM table-based question answering task.

evaluate_bimqa_xlnet.ipynb
Jupyter notebook for evaluating the XLNet-based intent recognition model on the BIM query dataset.

Datasets

full_data.xlsx
Complete annotated BIM natural language query dataset, including query texts, intent labels, and corresponding table references.

val_data_all.csv
Validation dataset used for model evaluation, containing queries and ground-truth annotations.

Table databases.zip
Compressed archive of BIM sub-databases extracted from IFC models and converted into tabular format for table-based question answering.

Source Code

intent_classification_model.py
Python implementation of the XLNet-based intent recognition model.

tableQA_model.py
Python implementation of the TAPAS-based table question answering model.
