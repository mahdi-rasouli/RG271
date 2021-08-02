import io
from pathlib import Path

import boto3
import pandas as pd
from flask import Flask, render_template
from RG271.binary_cls.inference import InferenceModel

# Get test data
s3 = boto3.client("s3")
obj = s3.get_object(Bucket="ds-rg271", Key="data/labelled/stratified_random_sampling/test.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

# Get model
if not Path("model.pkl").exists():
    s3.download_file("ds-rg271", "models/stratified_random_sampling/model.pkl", "model.pkl")

model = InferenceModel("model.pkl")

app = Flask(__name__)


@app.route("/")
def index():
    row = df.sample(n=1)
    output = model.predict(row.iloc[0]["content"])

    return render_template("index.html", tweet_url=row.iloc[0]["url"], result=output)


if __name__ == "__main__":
    app.run(debug=True)
