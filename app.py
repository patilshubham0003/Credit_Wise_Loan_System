import streamlit as st
import numpy as np
import pickle as pkl

# load models
with open("model.pkl","rb") as f:
    model=pkl.load(f)
    
with open("onehotEncoder.pkl","rb") as f :
    OHE = pkl.load(f)
    
with open("scaler.pkl","rb") as f :
    scaler = pkl.load(f)
    


st.title("Credit Wise Loan System")    

Applicant_Income = st.number_input("Applicant Income",500,20000,3000)
Coapplicant_Income = st.number_input("Coapplicant Income",500,20000,3000)
Age = st.number_input("Age",20,60,30)
Dependents = st.number_input("Dependent",0,5,2)
Credit_Score = st.number_input("Credit Score",300,850,500)
Existing_Loans = st.number_input("Existing Loans",0,5)
DTI_Ratio = st.number_input("DTI Ratio(in percentage %)",0,100,50)/100
Savings = st.number_input("Savings",0,30000,15000)
Collateral_Value = st.number_input("Collateral Value",30,50000)
Loan_Amount = st.number_input("Loan Amount",1000,40000,20000)
Loan_Term = st.selectbox("Loan Term(In months)",[48.0,84.0,12.0,24.0,60.0,72.0,36.0])
Education_Level = st.selectbox("Education Level",[0,1],format_func=lambda x : "Graduate" if x==0 else "Under_graduate")
Employment_Status = st.selectbox("Employment Status",["Salaried","Self-employed","Unemployed","Contract"])#
Marital_Status = st.selectbox("Marital Status",["Married","Single"])
Loan_Purpose = st.selectbox("Loan Purpose",["Business","Car","Home","Education","Personal"])#
Property_Area = st.selectbox("Property Area",["Urban","Rural","Semiurban"])#
Gender = st.selectbox("Gender",["Male","Female"])
Employer_Category = st.selectbox("Employer Category",["Private","Government","MNC","Business","Unemployed"])#

# encoded data
ohe_data=OHE.transform([[Employment_Status,Marital_Status,Loan_Purpose,Property_Area,Gender,Employer_Category]])
enc_data=ohe_data[0]
inputs =np.array([Applicant_Income,Coapplicant_Income,Age,Dependents,Credit_Score,Existing_Loans,DTI_Ratio,Savings,Collateral_Value,Loan_Amount,Loan_Term,Education_Level])
all_inputs = np.concatenate((inputs,enc_data))

print(all_inputs[0])

# scale
scale_val = scaler.transform([all_inputs])

if st.button("Predict") :
    output = model.predict(scale_val)
    print(output)
    
    if output[0] == 0:
        st.success("The loan not approved")
    else:
        st.success("the loan approved")
         