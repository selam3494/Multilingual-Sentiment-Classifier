# sentiment/predict.py
import argparse
from .model_io import load_model, predict_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="text to classify")
    ap.add_argument("--model_dir", default="model_en_light_best")
    ap.add_argument("--thresh", type=float, default=0.6)
    args = ap.parse_args()

    tok, model, device = load_model(args.model_dir)
    res = predict_one(args.text, tok, model, device, args.thresh)
    print(res)

if __name__ == "__main__":
    main()
