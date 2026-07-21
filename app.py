import os
import joblib
import pandas as pd
import gradio as gr

# Load model
model = joblib.load("loan_prediction_model.pkl")


def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    # Convert categorical values
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    data = pd.DataFrame(
        [[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value,
        ]],
        columns=[
            " no_of_dependents",
            " education",
            " self_employed",
            " income_annum",
            " loan_amount",
            " loan_term",
            " cibil_score",
            " residential_assets_value",
            " commercial_assets_value",
            " luxury_assets_value",
            " bank_asset_value",
        ],
    )

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "✅ Loan Approved"
    else:
        return "❌ Loan Rejected"


interface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Radio(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏦 Loan Approval Prediction",
    description="Enter applicant details to predict loan approval.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)
