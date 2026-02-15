# Radiomics 10: Whole-Body Metastasis Detection

## Overview
This project establishes a generalized, whole-body framework for metastasis detection using multi-modal PET/CT imaging. By utilizing foundation models like MedSAM, the system performs comprehensive segmentation across the entire anatomy, enabling a holistic analysis that transcends the limitations of organ-specific models. The pipeline is designed to correlate whole-body imaging features with genetic mutations, providing a scalable and integrated diagnostic tool for oncology.

## Project Goals
Based on the project requirements, the system focuses on achieving the following benchmarks:
* **Segmentation Precision**: Achieving a Dice Coefficient > 0.85 on validation sets.
* **Metastasis Prediction**: Reaching a classification accuracy > 0.90 and an F1 Score > 0.85.
* **Biomarker Discovery**: Identifying at least 10 statistically validated radiomic features linked to metastasis or genetic mutations.

## Tech Stack
* **Programming**: Python.
* **Core Libraries**: NumPy, Pandas, PyRadiomics, and PyTorch.
* **Infrastructure**: Containerized microservices using Docker and Kubernetes.
* **Database**: MongoDB for metadata and feature storage.

## Current Status
* **Initial Validation**: A prediction model has been developed and preliminary testing has been conducted on a cohort of patients to validate the logic.
* **In Progress**: Scaling the pipeline for full whole-body analysis and fine-tuning MedSAM.

## Project Team
* **Students**: Asaf Zini & Omer Bar-Eli.
* **Supervisor**: Prof. Ilan Tsarfaty, Tel Aviv University.

