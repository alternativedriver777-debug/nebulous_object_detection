def train(data_path, epochs, batch, model_name):
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(data=data_path, epochs=epochs, batch=batch)
    print("Обучение завершено. Веса сохранены в runs/detect.")

