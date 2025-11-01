import modal
app = modal.App("gqa-comparison")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("pytorch-lightning==2.4.0")
    .run_commands(
        "pip install --upgrade 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    .pip_install("flax==0.10.2")
    .add_local_python_source("gqa_comparison")
)

@app.function(image=image)
def run_benchmarks() -> None:
    from gqa_comparison import main as run
    run()

@app.local_entrypoint()
def main(gpu: str = "A100-40GB", timeout: int = 3600) -> None:
    run_benchmarks.with_options(gpu=gpu, timeout=timeout).remote()
