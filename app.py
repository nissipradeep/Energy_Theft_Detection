import pandas as pd
import numpy as np
import joblib
import gradio as gr

# -----------------------------
# 1. Load Saved Artifacts
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# -----------------------------
# 2. Feature Definitions
# -----------------------------
NUM_COLS = ["Energy_Consumption", "Previous_Bills", "Average_Temperature"]
CAT_COLS = ["Location", "Time_of_Use", "Payment_Method", "Consumption_Type"]

# -----------------------------
# 3. Prediction Logic
# -----------------------------
def predict_energy_theft(
    age,
    energy_consumption,
    location,
    time_of_use,
    previous_bills,
    average_temperature,
    payment_method,
    consumption_type
):
    # Create input dataframe
    input_df = pd.DataFrame([{
        "Age": age,
        "Energy_Consumption": energy_consumption,
        "Previous_Bills": previous_bills,
        "Average_Temperature": average_temperature,
        "Location": location,
        "Time_of_Use": time_of_use,
        "Payment_Method": payment_method,
        "Consumption_Type": consumption_type
    }])

    # Scale numerical features
    input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

    # Encode categorical features
    encoded = encoder.transform(input_df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CAT_COLS)
    )

    # Combine final input
    final_input = pd.concat(
        [input_df[["Age"] + NUM_COLS].reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Predict probabilities
    probs = model.predict_proba(final_input)[0]
    abnormal_prob = probs[0]
    normal_prob = probs[1]

    # Decision threshold (strict)
    if abnormal_prob >= 0.5:
        return f"🚨 Abnormal Usage Detected (Theft Risk: {abnormal_prob:.2%})"
    else:
        return f"✅ Normal Usage (Confidence: {normal_prob:.2%})"

# -----------------------------
# 4. Gradio Interface
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ⚡ Energy Theft Detection System  
    **Machine Learning–based electricity theft identification**  
    Enter customer details to classify energy usage as **Normal** or **Abnormal**.
    """)

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=30)
            energy = gr.Number(label="Energy Consumption (kWh)", value=500)
            location = gr.Dropdown(
                ["Urban", "Rural", "Suburban"],
                label="Location",
                value="Urban"
            )
            time_of_use = gr.Dropdown(
                ["Day", "Night"],
                label="Time of Use",
                value="Day"
            )
            bills = gr.Number(label="Previous Bills ($)", value=200)
            temperature = gr.Number(label="Average Temperature (°C)", value=25)
            payment = gr.Dropdown(
                ["Cash", "Credit Card", "Debit Card"],
                label="Payment Method",
                value="Cash"
            )
            ctype = gr.Dropdown(
                ["Residential", "Commercial"],
                label="Consumption Type",
                value="Residential"
            )

            btn = gr.Button("Analyze Usage", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Prediction Result")

    btn.click(
        fn=predict_energy_theft,
        inputs=[
            age,
            energy,
            location,
            time_of_use,
            bills,
            temperature,
            payment,
            ctype
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            [35, 1500, "Urban", "Night", 50, 32, "Cash", "Residential"],
            [40, 220, "Suburban", "Day", 240, 25, "Credit Card", "Residential"]
        ],
        inputs=[
            age,
            energy,
            location,
            time_of_use,
            bills,
            temperature,
            payment,
            ctype
        ]
    )

# -----------------------------
# 5. Launch App
# -----------------------------
demo.launch()
