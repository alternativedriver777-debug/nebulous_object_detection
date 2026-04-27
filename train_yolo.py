import argparse

from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.yaml", help="path to data yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch)

    print("Обучение завершено. Веса сохранены в runs/detect")
