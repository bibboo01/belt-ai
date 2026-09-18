from app.database.repository import DatabaseRepository


def main():
    print("=" * 70)
    print("BELT AI - MODEL REGISTRY TEST")
    print("=" * 70)

    repository = DatabaseRepository()

    model = repository.get_model_version(
        model_name="YOLO11",
        model_version="ODBv54",
    )

    if model is None:
        print("Model found       : False")
        print("Status             : FAILED")
        return

    print("Model found       : True")
    print(f"Model ID          : {model['model_id']}")
    print(f"Model Name        : {model['model_name']}")
    print(f"Model Version     : {model['model_version']}")
    print(f"Model Type        : {model['model_type']}")
    print(f"Model Path        : {model['model_path']}")
    print(f"Status            : {model['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()