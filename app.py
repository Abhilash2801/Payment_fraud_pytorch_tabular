from flask import Flask, request, jsonify
import pandas as pd
import os
import pytorch_tabular
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel
from pytorch_tabular.models import AutoIntConfig
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score,accuracy_score ,f1_score, roc_auc_score
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the CSV file
            df = pd.read_csv(file_path)
            from sklearn.model_selection import train_test_split
            train_dat, test_dat = train_test_split(df, random_state=42, test_size=0.2)
            train_dat, val = train_test_split(train_dat, random_state=42, test_size=0.2)
            # Perform any additional actions on the DataFrame if needed
            cat_col_names=df.select_dtypes(include=["object"]).columns.tolist()
            num_col_names=df.select_dtypes(include=["int64","float64"]).columns.tolist()
            num_col_names.remove("is_fraud")

            data_config = DataConfig(
                target=["is_fraud"],
                continuous_cols=num_col_names,
                categorical_cols=cat_col_names
            )
            optimiser_config=OptimizerConfig()

            trainer_config = TrainerConfig(
                batch_size=1024,  # Batch size for training
                max_epochs=100,  # Maximum number of epochs
            )
            model_config_3=AutoIntConfig(
                task="classification",
            )

            tabular_model_3=TabularModel(
                data_config=data_config,
                trainer_config=trainer_config,
                optimizer_config=optimiser_config,
                model_config=model_config_3,
                verbose=True
            )
            modellist=["Autoint","Category","Danet","Gandalf","gate","Tabnet","TabTrans"]
            frame=[]
            for i in modellist:
                tabular_model_3= TabularModel.load_model(i)
                x=tabular_model_3.predict(test_dat)
                true_labels=test_dat["is_fraud"].values
                pred_labels=x['prediction'].astype(int)
                accuracy=accuracy_score(true_labels, pred_labels)
                f1score= f1_score(true_labels, pred_labels)
                frame.append([i,accuracy,f1score])
            daf=pd.DataFrame(frame,columns=["model","accuracy","f1_score"])
            # Convert DataFrame to JSON and return
            return daf.to_json(orient='records')
    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    app.run(debug=True, port=8080)
