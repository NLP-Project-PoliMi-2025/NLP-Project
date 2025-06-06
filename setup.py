from setuptools import setup

setup(
    name='nlpChess',
    version='0.1.0',
    description='A packege for NLP and Chess',
    url='https://github.com/NLP-Project-PoliMi-2025/NLP-Project',
    author='Paolo Ginefra, Ferdinando Onori, Martina Missana, Robin Uhrich',
    author_email='',
    license='MIT License',
    packages=['nlpChess'],
    install_requires=[
        "numpy (>=1.18.5,<2.0.0)",
        "matplotlib (>=3.10.1,<4.0.0)",
        "jupyter (>=1.1.1,<2.0.0)",
        "tqdm (>=4.67.1,<5.0.0)",
        "polars (>=1.28.1,<2.0.0)",
        "mlcroissant (>=1.0.17,<2.0.0)",
        "datasets (>=3.5.1,<4.0.0)",
        "huggingface-hub[hf-xet] (>=0.30.2,<0.31.0)",
        "chess (>=1.11.2,<2.0.0)",
        "pydot (>=4.0.0,<5.0.0)",
        "torch (>=2.7.0,<3.0.0)",
        "lightning (>=2.5.1.post0,<3.0.0)",
        "nltk (>=3.9.1,<4.0.0)",
        "gensim (>=4.3.3,<5.0.0)",
        "einops (>=0.8.1,<0.9.0)",
        "pyargwriter (>=1.1.3,<2.0.0)",
        "pytest (>=8.3.5,<9.0.0)",
        "lark (>=1.2.2,<2.0.0)",
        "scikit-learn (>=1.6.1,<2.0.0)",
        "seaborn (>=0.13.2,<0.14.0)",
        "cairosvg (>=2.8.2,<3.0.0)",
        "pygame (>=2.6.1,<3.0.0)",
        "transformers (>=4.51.3,<5.0.0)",
        "tensorboard (>=2.19.0,<3.0.0)",
        "wandb (>=0.19.11,<0.20.0)",
        "bitsandbytes (>=0.45.5,<0.46.0)",
        "accelerate (>=1.7.0,<2.0.0)",
        "peft (>=0.15.2,<0.16.0)",
        "ffmpeg (>=1.4,<2.0)",
        "sounddevice (>=0.5.2,<0.6.0)",
        "beepy (>=1.0.7,<2.0.0)",
        "librosa (>=0.11.0,<0.12.0)"
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
