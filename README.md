# Mechanism-Aware-Hybrid-Physics-XGBoost-Model-for-Shear-Strength-of-FRP-Strengthened-RC-Beams
ML-based prediction of shear strength in FRP-strengthened concrete beams with explainable AI (SHAP).

## Overview
This project presents a data-driven framework for predicting the shear strength of FRP-strengthened reinforced concrete beams using machine learning models and explainable AI techniques.

## Objectives
- Develop accurate ML models for shear strength prediction
- Compare multiple algorithms
- Interpret model predictions using SHAP

## Models Used
- XGBoost (Primary Model)
- LightGBM
- Random Forest
- Linear Regression

## Features
- Geometric parameters (b, d, a/d)
- Material properties (fc, fy)
- FRP parameters (Ef, rho_f, thickness, angle)

## Target
- Shear Strength (Vu)

## Installation
```bash
pip install -r requirements.txt
