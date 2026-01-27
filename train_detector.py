from ultralytics import YOLO


def main():
    model = YOLO('yolov8s.pt')


    print("Rozpoczynam trening detektora tablic...")

    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        plots=True
    )

    print("\nRozpoczynam walidację na zbiorze testowym...")
    metrics = model.val()

    print("\n--- WYNIKI ---")
    print(f"mAP50 (skuteczność przy IoU > 0.5): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (średnia z różnych progów IoU): {metrics.box.map:.4f}")

    path = model.export(format='onnx')
    print(f"Model wyeksportowany do: {path}")


if __name__ == '__main__':
    main()