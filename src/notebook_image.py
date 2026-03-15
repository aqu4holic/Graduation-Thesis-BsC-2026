import modal

# Image definition here:
image = (
    modal.Image.from_registry("python:3.13-slim")
    .uv_sync(
        uv_project_dir="../",
        gpu="A100-40GB"
    )
    .add_local_dir(
        local_path="./",
        remote_path="/mnt/src/"
    )
)

app = modal.App("notebook-images")

@app.function(image=image)  # You need a Function object to reference the image.
def notebook_image():
    pass