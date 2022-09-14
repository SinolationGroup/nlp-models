from time import time

import pandas as pd
import streamlit as st
import torch

from src.models.predict_model import OHLProductionModel

MODEL_ID = "models/last.ckpt"
DEVICE = "cpu"

st.set_page_config(
    page_title="Demo",
    page_icon=":hugging_face:",
)

st.header("Demo")

# init model
@st.cache(show_spinner=True, allow_output_mutation=False)
def create_model(device: str, model_id: str):

    model = OHLProductionModel(n_classes=296, device=device)
    state_dict = torch.load(model_id, map_location=torch.device(model.device))[
        "state_dict"
    ]
    model.load_state_dict(state_dict)
    model.eval().to(model.device)
    model.warmup()
    return model


model = create_model(device=DEVICE, model_id=MODEL_ID)


@st.cache(show_spinner=True, allow_output_mutation=False)
def create_data():
    df = pd.read_csv("data/raw/NLP training data 0816.csv")
    df.drop_duplicates(inplace=True)
    df.reset_index(
        inplace=True,
    )

    class_names = []
    with open("data/interim/class_names.txt", "r") as f:
        for line in f:
            class_names.append(line.strip())

    return df, class_names


df, class_names = create_data()

question = st.text_input(
    "Type your question and click Predict!", "i want to speak to a manager"
)
predict_btn = st.button("Predict!")
pred_table = st.empty()

if predict_btn:

    if question:
        start = time()
        input = model.preprocess_input(question)
        pred = model(input).cpu().detach().numpy()
        time_range = time() - start

        idx = pred.argsort(axis=1)
        preds = []
        for e in idx[0, -5:][::-1]:
            temp_dict = {}
            temp_dict["class_name"] = class_names[e]
            temp_dict["p"] = f"{pred[0, e]:.2%}"
            temp_dict["q"] = df[df.question_id == temp_dict["class_name"]][
                "question_description"
            ].values[0]
            preds.append(temp_dict)
        preds_df = pd.DataFrame(preds)
        pred_table.dataframe(preds_df)

        time_msec = round(time_range * 1000, 2)
        st.write(f"Inference time (CPU): {time_msec} msec")
