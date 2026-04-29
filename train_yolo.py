import argparse

from nebulous_detector.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.yaml", help="path to data yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    train(data_path=args.data, epochs=args.epochs, batch=args.batch, model_name=args.model)


if __name__ == "__main__":
    main()

