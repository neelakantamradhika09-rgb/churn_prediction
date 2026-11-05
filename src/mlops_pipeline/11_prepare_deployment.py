
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import yaml
import skops.io as sio
from mlflow import MlflowClient


MODEL_NAME = "Telco_Churn_Production_Model"


def load_config():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return project_root, config


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(
            f"No Production version found for '{model_name}'."
        )

    if len(production_versions) > 1:
        raise RuntimeError(
            f"Multiple Production versions found for '{model_name}'."
        )

    return production_versions[0]


def load_trusted_model(model_uri):
    print("[INFO] Downloading model from MLflow...")

    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )

    model_dir = Path(local_model_path)

    mlmodel_file = model_dir / "MLmodel"

    if not mlmodel_file.exists():
        raise FileNotFoundError(
            f"MLmodel file not found in {model_dir}"
        )

    with open(mlmodel_file, "r", encoding="utf-8") as file:
        model_config = yaml.safe_load(file)

    sklearn_config = model_config.get("flavors", {}).get("sklearn")

    if not sklearn_config:
        raise RuntimeError("Scikit-learn flavor not found in MLflow model.")

    data_file = model_dir / sklearn_config.get("data", "model.pkl")

    if not data_file.exists():
        raise FileNotFoundError(
            f"Model data file not found: {data_file}"
        )

    serialization_format = sklearn_config.get(
        "serialization_format", "pickle"
    )

    print(f"[INFO] Serialization format: {serialization_format}")

    if serialization_format == "skops":
        print("[INFO] Loading trusted Skops model...")

        model = sio.load(
            data_file,
            trusted=["sklearn.tree._tree.Tree"]
        )

    else:
        print("[INFO] Loading model using MLflow...")

        model = mlflow.sklearn.load_model(str(model_dir))

    return model


def prepare_deployment():

    project_root, config = load_config()

    deployment_dir = project_root / "deployment"

    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)

    deployment_dir.mkdir(parents=True)

    client = MlflowClient()

    model_version = get_production_model(
        client,
        MODEL_NAME
    )

    model_name = model_version.name
    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("\n[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")
    print(f"[INFO] Source     : {model_source}")

    print("\n[INFO] Downloading Production model...")

    model_uri = f"models:/{model_name}/{version}"

    model = load_trusted_model(model_uri)

    model_path = deployment_dir / "model.pkl"

    joblib.dump(model, model_path)

    print(f"[SUCCESS] Model saved to {model_path}")

    print("\n[INFO] Downloading matching preprocessor...")

    preprocessor_path = client.download_artifacts(
        run_id,
        "preprocessing_pipeline/preprocessor.pkl",
        dst_path=str(deployment_dir),
    )

    downloaded_preprocessor = Path(preprocessor_path)

    final_preprocessor_path = deployment_dir / "preprocessor.pkl"

    if downloaded_preprocessor.resolve() != final_preprocessor_path.resolve():
        shutil.move(
            str(downloaded_preprocessor),
            str(final_preprocessor_path)
        )

    print(
        f"[SUCCESS] Preprocessor saved to {final_preprocessor_path}"
    )

    print("\n[INFO] Verifying deployment artifacts...")

    loaded_model = joblib.load(model_path)

    loaded_preprocessor = joblib.load(
        final_preprocessor_path
    )

    if loaded_model is None:
        raise RuntimeError("Model verification failed.")

    if loaded_preprocessor is None:
        raise RuntimeError("Preprocessor verification failed.")

    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": model_version.current_stage,
        "run_id": run_id,
        "model_source": model_source,
        "model_id": model_source.split("/")[-1],
        "preprocessor_artifact":
            "preprocessing_pipeline/preprocessor.pkl",
        "source": "MLflow"
    }

    metadata_path = deployment_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Metadata saved to {metadata_path}")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("├── model.pkl")
    print("├── preprocessor.pkl")
    print("└── model_metadata.json")

    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    prepare_deployment()
