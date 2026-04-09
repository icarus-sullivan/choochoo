from setuptools import setup, find_packages

setup(
    name="choochoo",
    version="0.1.0",
    description="High-performance LoRA training framework for diffusion and video models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "diffusers>=0.27.0",
        "peft>=0.10.0",
        "safetensors>=0.4.3",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "pillow>=10.3.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "psutil>=5.9.8",
    ],
    entry_points={
        "console_scripts": [
            "choochoo-train=train:main",
        ],
    },
)
