import modal
from pathlib import Path

VOLUME_NAME = "thesis"
LOCAL_DATASET_DIR = Path("dataset_cache")
REMOTE_DATASET_DIR = "/dataset_cache"

# Creates the volume if it doesn't already exist
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

def main():
    files = list(LOCAL_DATASET_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]

    if not files:
        print(f"No files found in {LOCAL_DATASET_DIR}/")
        return

    print(f"Uploading {len(files)} files to volume '{VOLUME_NAME}'...")

    with volume.batch_upload(force=True) as upload:
        for local_path in files:
            # Preserve the directory structure inside the volume
            remote_path = REMOTE_DATASET_DIR / local_path.relative_to(LOCAL_DATASET_DIR)
            upload.put_file(local_path, remote_path.as_posix())
            print(f"  {local_path} → {remote_path}")

    print("Done! Dataset uploaded successfully.")

if __name__ == "__main__":
    main()