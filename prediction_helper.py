from joblib import load
import pandas as pd
import os
from joblib import load

model_rest_path = os.path.join("artifacts", "model_rest.joblib")
model_rest = load(model_rest_path)

model_young_path = os.path.join("artifacts", "model_young.joblib")
model_young = load(model_young_path)

scaler_rest_path = os.path.join("artifacts", "scaler_rest.joblib")
scaler_rest = load(scaler_rest_path)

scaler_young_path = os.path.join("artifacts", "scaler_young.joblib")
scaler_young = load(scaler_young_path)


#model_rest = load("artifacts\model_rest.joblib")
#model_young = load("artifacts\model_young.joblib")

#scaler_rest = load("artifacts\scaler_rest.joblib")
#scaler_young = load("artifacts\scaler_young.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {'diabetes': 6,
     'high blood pressure': 6,
     'heart disease': 8,
     'thyroid': 5,
     'no disease': 0,
     'none': 0}

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease,0) for disease in diseases) #default to 0 if disease not found

    max_score=14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high bp
    min_score=0

    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score

###############################################

def preprocess_input(input_dict):
    expected_columns=[
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
       'genetical_risk', 'normalized_risk_score', 'gender_Male',
       'region_Northwest', 'region_Southeast', 'region_Southwest',
       'marital_status_Unmarried', 'bmi_category_Obesity',
       'bmi_category_Overweight', 'bmi_category_Underweight',
       'smoking_status_Occasional', 'smoking_status_Regular',
       'employment_status_Salaried', 'employment_status_Self-Employed'
        ]

    insurance_plan_encoding = {"Bronze" : 1, "Silver" : 2, "Gold" : 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    for key,value in input_dict.items():
        if key == "Age":
            df['age'] = value
        elif key == "Number of Dependants":
            df['number_of_dependants'] = value
        elif key == "Income in Lakhs":
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value
        elif key == "Insurance Plan":
            df['insurance_plan'] = insurance_plan_encoding.get(value,1)
        elif key == "Gender" and value == "Male":
            df['gender_Male'] = 1
        elif key == "Region":
            if value == "Northwest":
                df['region_Northwest'] = 1
            elif value == "Southeast":
                df['region_Southeast'] = 1
            elif value == "Southwest":
                df['region_Southewest'] = 1
        elif key == "Marital Status" and value == "Unmarried":
            df['marital_status_Unmarried'] = 1
        elif (key == "BMI Category"):
            if value == "Obesity":
                df["bmi_category_Obesity"] = 1
            elif value == "Overweight":
                df["bmi_category_Overweight"] = 1
            elif value == "Underweight":
                df["bmi_category_Underweight"] = 1
        elif key == "Smoking Status":
            if value == "Regular":
                df["smoking_status_Regular"] = 1
            elif value == "Occasional":
                df["smoking_status_Occasional"] = 1
        elif key == "Employment Status":
            if value == "Salaried" :
                df["employment_status_Salaried"] = 1
            elif value == "Self-Employed":
                df["employment_status_Self-Employed"] = 1

    df["normalized_risk_score"] = calculate_normalized_risk(input_dict['Medical History'])

    df = handle_scaling(input_dict["Age"], df)
    return df
#################################################
def handle_scaling(age, df):
    if age<=25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]
    df["income_level"] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop("income_level", axis = "columns", inplace=True)
    return df
##############################
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    if input_dict["Age"] <=25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction)
