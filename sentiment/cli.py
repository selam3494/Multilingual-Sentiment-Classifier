import argparse
import json

from .trainer import train_model
from .evaluate import eval_model
from .model_io import load_model, predict_one

def main():
    parser = argparse.ArgumentParser(prog="sentiment-cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # TRAIN
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--lang", default="en")
    p_train.add_argument("--model_name", default=None)
    p_train.add_argument("--per_class_train", type=int, default=2000)
    p_train.add_argument("--per_class_val",   type=int, default=400)
    p_train.add_argument("--per_class_test",  type=int, default=600)
    p_train.add_argument("--output_dir", default="model_en_light_best")
    p_train.add_argument("--cpu", action="store_true")

    # EVAL
    p_eval = sub.add_parser("eval", help="Evaluate a model")
    p_eval.add_argument("--model_dir", default="model_en_light_best")
    p_eval.add_argument("--lang", default="en")
    p_eval.add_argument("--n_test", type=int, default=None)
    p_eval.add_argument("--cm_path", default="confusion_matrix.png")

    # PREDICT
    p_pred = sub.add_parser("predict", help="Predict one text")
    p_pred.add_argument("text")
    p_pred.add_argument("--model_dir", default="model_en_light_best")
    p_pred.add_argument("--thresh", type=float, default=0.6)

    args = parser.parse_args()

    if args.cmd == "train":
        save_dir, best = train_model(
            lang=args.lang,
            model_name=args.model_name or "distilbert-base-multilingual-cased",
            per_class_train=args.per_class_train,
            per_class_val=args.per_class_val,
            per_class_test=args.per_class_test,
            output_dir=args.output_dir,
            use_cuda=not args.cpu
        )
        print(f"Best model saved to {save_dir} with best macro F1 {best:.4f}")

    elif args.cmd == "eval":
        macro, cm_path = eval_model(
            model_dir=args.model_dir,
            lang=args.lang,
            n_test=args.n_test,
            cm_path=args.cm_path
        )
        print(f"Macro F1: {macro:.4f}")

    elif args.cmd == "predict":
        tok, model, device = load_model(args.model_dir)
        res = predict_one(args.text, tok, model, device, args.thresh)
        print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
