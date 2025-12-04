"""
MLFlow utility functions for model management
"""
import mlflow
import torch
from typing import Optional, Dict


def register_model(run_id: str, model_name: str = "face_attributes_multihead") -> str:
    """
    Register a model from a specific run to the MLFlow Model Registry
    
    Args:
        run_id: MLFlow run ID containing the model
        model_name: Name for the registered model
        
    Returns:
        Model version as a string
    """
    client = mlflow.tracking.MlflowClient()

    # Get model URI from run
    model_uri = f"runs:/{run_id}/model"

    # Register the model
    model_details = mlflow.register_model(model_uri, model_name)

    print(f" Model registered: {model_name} version {model_details.version}")
    return str(model_details.version)


def load_best_model_from_mlflow(
    model_name: str = "face_attributes_multihead",
    stage: str = "Production",
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load the best model from MLFlow Model Registry
    
    Args:
        model_name: Name of the registered model
        stage: Stage to load from (Staging, Production, None)
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    # Construct model URI
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Load latest version
        model_uri = f"models:/{model_name}/latest"

    print(f" Loading model from MLFlow: {model_uri}")

    # Load the model
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()

    print(f" Model loaded successfully!")
    return model


def promote_model(
    model_name: str = "face_attributes_multihead",
    version: Optional[str] = None,
    stage: str = "Production"
) -> None:
    """
    Promote a model version to a specific stage (Staging → Production)
    
    Args:
        model_name: Name of the registered model
        version: Model version to promote (if None, uses latest)
        stage: Target stage (Staging, Production, Archived)
    """
    client = mlflow.tracking.MlflowClient()

    # If no version specified, get the latest
    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        version = max(int(v.version) for v in versions)

    print(f" Promoting model '{model_name}' version {version} to {stage}...")

    # Transition model to new stage
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=True  # Archive old versions in this stage
    )

    print(f" Model version {version} promoted to {stage}!")


def get_model_info(model_name: str = "face_attributes_multihead") -> Dict:
    """
    Get information about all versions of a registered model
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        Dictionary with model version information
    """
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        return {"model_name": model_name, "versions": []}

    version_info = []
    for v in versions:
        version_info.append({
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "creation_timestamp": v.creation_timestamp,
            "last_updated_timestamp": v.last_updated_timestamp
        })

    # Sort by version number
    version_info.sort(key=lambda x: int(x["version"]), reverse=True)

    return {
        "model_name": model_name,
        "total_versions": len(version_info),
        "versions": version_info
    }


if __name__ == "__main__":
    # Example usage
    print("MLFlow Utils - Example Usage")
    print("=" * 60)

    # Get model info
    try:
        info = get_model_info()
        print(f"\nModel: {info['model_name']}")
        print(f"Total versions: {info['total_versions']}")

        if info['versions']:
            print("\nVersions:")
            for v in info['versions']:
                print(f"  - Version {v['version']}: {v['stage']}")
    except Exception as e:
        print(f"Note: {e}")
        print("Run a training first to register models!")