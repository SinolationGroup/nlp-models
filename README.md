# Question Matching

We need to train a model to classify user questions. We must ensure low latency. The dataset includes many imbalanced classes

## Model

We used architecture “transformers” that are state of the art in nlp. We used distilled versions of the models that provide the lowest latency. We have found that the model “distilbert” is the most suitable model for our task.

We used Transformers and Pytorch.  During the training, we used Pytorch Lightning which is a Pytorch wrapper that simplifies creating training and validation pipelines. To increase the diversity of the data, we added data augmentation using NLPaug library.

## Environment

- Ubuntu 20.04
- Python 3.8
- GPU RTX 3060 12GB
- CUDA 11.3
- Libraries in `requirements.txt` file. Run

    ```bash
    virtualenv .env 
    source .env/bin/activate
    pip install -r requirements.txt
    ```

## Training

### Steps to run training

1. Copy `NLP training data 0816.csv` to `data/raw/` folder.
2. Run

    ```bash
    python -m src.data.make_dataset "data/raw/NLP training data 0816.csv" "data/external/paraphrased_questions.xlsx" "data/processed/dataset.csv"
    ```

    This script will generate a dataset
3. Run training script

    ```bash
    python -m src.models.train_model
    ```

    To track experiments we use wandb. It allows us to track training metrics and parameters using web UI. The first time you start the training, you will need to login to wandb. If you don’t want to use wandb then comment the line
    `logger=wandb_logger`

    You can find trained models in `models/chkpts folder`

### Train parameters

You can change them in the file `src/config.py`

```python
MODEL_NAME = "distilbert-base-uncased"
LR_TOP = 1e-4
PATH_DATASET = "data/processed/dataset.csv"
MAX_LENGTH = None # we use dynamic sequence length
TEST_SIZE = 0.3
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
N_EPOCHS = 3
```

### To train on new data

You need to add new questions and labels to `data/processed/dataset.csv` and run training again. You can use a text editor or excel for this.

## Inference pipeline

Inference dependencies only include torch and transformers libraries. Code of inference model can be found in `src/model/predict_model.py` file

```bash
pip3 install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.21.1
```

1. Import model code

    ```python
    from src.models.predict_model import OHLProductionModel
    ```

2. Define constants: checkpoint file path (file with model parameters) and device (“cpu” or “cuda”) on which the calculations will be performed

    ```python
    MODEL_ID = "models/model.ckpt"
    DEVICE = "cpu"
    ```

3. Create a model and load trained parameters

    ```python
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
    ```

4. Preprocess input. You can feed one text

    ```python
    input = model.preprocess_input(“test text”)
    ```

    or several texts

    ```python
    input = model.preprocess_input([“test text”, “test text text”])
    ```

5. Make prediction

    ```python
    pred = model(input).cpu().detach().numpy()
    ```

    pred is a numpy array with shape [1, 296] for one text or [n, 296] for n texts. Each number in this array is the probability that the text is associated with a particular class.
6. Load list of class names

    ```python
    class_names = []
        with open("data/interim/class_names.txt", "r") as f:
            for line in f:
                class_names.append(line.strip())
    ```

7. Get class name

    ```python
    idx = pred.argsort[axis=1]
    pred_class_name = class_names[idx[0][::-1][0]] # the class with the highest probability for the first text
    pred_class_name = class_names[idx[1][::-1][0]] # for the second text
    ```

## Demo

You can find an example of using the model in the file `app/demo.py`. First you need to copy file `last.ckpt` to the folder `models`. To run demo execute command and open <http://localhost:8501> in browser

```bash
streamlit run app/demo.py
```

Also there is an example in `notebooks/08-predict.ipynb` file
