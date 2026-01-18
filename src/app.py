import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/model.pkl")

# Prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(data)[0]

    species_map = {
        0: "Iris setosa",
        1: "Iris versicolor",
        2: "Iris virginica"
    }

    return species_map[pred]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌸 Iris Flower Classifier")
    gr.Markdown("Enter the flower measurements to predict the species.")

    with gr.Row():
        sepal_length = gr.Number(label="Sepal Length (cm)")
        sepal_width = gr.Number(label="Sepal Width (cm)")
        petal_length = gr.Number(label="Petal Length (cm)")
        petal_width = gr.Number(label="Petal Width (cm)")

    predict_btn = gr.Button("Predict Species")
    output = gr.Textbox(label="Prediction")

    predict_btn.click(
        fn=predict_species,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=output
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
