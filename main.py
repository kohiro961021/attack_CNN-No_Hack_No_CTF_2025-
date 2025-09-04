from flask import Flask, request, render_template
from predict import load_models, run_inference
import os
import uuid

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = "static"

model_v8, model_v10 = load_models() 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            
            filename = f"{uuid.uuid4().hex}.png"

            
            raw_path = os.path.join(STATIC_DIR, filename)
            v8_path = os.path.join(STATIC_DIR, f"v8_{filename}")
            v10_path = os.path.join(STATIC_DIR, f"v10_{filename}")
            file.save(raw_path)

            
            result_v8 = run_inference(model_v8, raw_path, v8_path)
            result_v10 = run_inference(model_v10, raw_path, v10_path)

            
            different_prediction = result_v8["class_name"] != result_v10["class_name"]
            confidence_gap = abs(result_v8["confidence"] - result_v10["confidence"]) >= 0.4

            if different_prediction or confidence_gap:
                with open(os.path.join(APP_ROOT, "../flag.txt")) as f:
                    flag = f.read().strip()
                return render_template("result.html", flag=flag,
                                       v8=result_v8, v10=result_v10,
                                       v8_path=f"static/v8_{filename}",
                                       v10_path=f"static/v10_{filename}")
            else:
                return render_template("result.html", flag=None,
                                       v8=result_v8, v10=result_v10,
                                       v8_path=f"static/v8_{filename}",
                                       v10_path=f"static/v10_{filename}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
