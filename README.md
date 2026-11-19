# Sketches to Life

Turn a hand-drawn sketch into a photorealistic image with **Stable Diffusion 1.5** and **ControlNet**. The project is a single notebook that loads sketches, cleans them into a conditioning map, generates images, sweeps ControlNet strength and text guidance, and scores how well the output still follows the original lines.

Run it top to bottom on a GPU (Kaggle or a local CUDA machine). Metrics in the notebook are computed from that run. Nothing in the notebook fabricates results.

## What it does

1. Uses local sketch images when you provide them. If fewer than three are present, it downloads recognized stroke drawings from Google’s [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset (`butterfly`, `camera`, and `house` as a simple building) and renders them as black-on-white PNGs.
2. Resizes each sketch to 512×512, corrects dark or light backgrounds, and cleans strokes with adaptive thresholding, small-component removal, and optional thickening.
3. Generates an image with Stable Diffusion 1.5 conditioned by a scribble or line-art ControlNet.
4. Runs two one-at-a-time sweeps on up to three sketches: ControlNet scale and guidance scale.
5. Measures sketch preservation with edge recall and Chamfer distance, then writes images, CSVs, a run manifest, and a results ZIP.
6. Includes an optional Gradio demo so you can upload a sketch and change strength, guidance, steps, and seed.

## Models

| Role | Checkpoint |
| --- | --- |
| Base model | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Scribble ControlNet (default) | [`lllyasviel/control_v11p_sd15_scribble`](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) |
| Line-art ControlNet | [`lllyasviel/control_v11p_sd15_lineart`](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) |
| Scheduler | UniPC multistep |

Weights download from Hugging Face on the first run. Internet access is required. On CUDA the pipeline uses float16 and attention slicing. On CPU it falls back to float32, which is too slow for a full experiment.

Stable Diffusion 1.5 is released under the [CreativeML Open RAIL-M license](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE.md).

## Requirements

- Python 3.10 or newer
- A CUDA GPU with enough memory for Stable Diffusion 1.5 (about 8 GB is a practical minimum with float16 and attention slicing)
- Internet access for model and dataset downloads
- Packages in [`requirements.txt`](requirements.txt)

PyTorch is installed separately so you can pick the build that matches your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

On Kaggle or Colab, use the environment’s existing PyTorch and run the install cell in the notebook. The Gradio cell is optional; `gradio` is listed in `requirements.txt` for that demo.

## How to run

Open [`Sketches to Life.ipynb`](Sketches%20to%20Life.ipynb) and run every cell from top to bottom.

**Kaggle**

1. Upload the notebook.
2. Set **Settings → Accelerator → GPU**.
3. Turn internet access on.
4. Run all cells.

**Local**

```bash
jupyter notebook "Sketches to Life.ipynb"
```

Outputs are written to `/kaggle/working/sketch2image` on Kaggle, or to `./sketch2image` when that path does not exist. That folder is gitignored.

### Use your own sketches

Put at least three images in `sketch2image/sketches/original/` before the input cell, or set `DATASET_DIR` in that cell to a folder of images. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.jfif`, and `.webp`. Local files replace the automatic Quick, Draw! downloads.

Name files to match a prompt key when you can: `butterfly`, `camera`, `building`, `city`, `eye`, `pencil`, `person`, `skull`, `station`, `tree`, `yellow_flowers`. Any other filename gets a generic prompt from the file stem. Edit the `PROMPTS` dictionary in the notebook to change the text.

### Main settings

| Setting | Default | Where to change it |
| --- | --- | --- |
| Control type | `scribble` (`lineart` is the other option) | `CONTROL_TYPE` |
| Inference steps | 28 | `BASE_STEPS` |
| Guidance scale | 7.0 | `BASE_GUIDANCE` |
| ControlNet scale | 1.0 | `BASE_CONTROL_SCALE` |
| Seed | 42 | `SEED` |
| Image size | 512 | `IMAGE_SIZE` |
| Negative prompt | blurry, low quality, deformed, distorted, watermark, text, oversaturated | `NEGATIVE_PROMPT` |

The parameter sweep uses ControlNet scales `0.5, 0.8, 1.0, 1.2, 1.5` and guidance scales `3.5, 5.0, 7.0, 9.0, 11.0` on the first three sketches. Set `EXPERIMENT_SKETCHES` to a list of names, or to `None` to keep that default.

## Notebook map

| Section | What it does |
| --- | --- |
| 1. Environment | Fixes seeds, creates folders, installs packages |
| 2. Inputs and prompts | Loads local sketches or Quick, Draw! strokes and assigns prompts |
| 3. Preprocessing | Builds the ControlNet conditioning map |
| 4. Pipeline | Loads Stable Diffusion 1.5 + ControlNet |
| 5. Baseline | Generates one image per sketch and saves JSON sidecars |
| 6. Experiments | Sweeps ControlNet scale and guidance scale |
| 7. Evaluation | Edge recall and Chamfer distance |
| 8. Interpretation | Prints run summaries and lists limitations |
| 9. Export | Writes `run_manifest.json` and `sketch2image_results.zip` |
| 10. Gradio | Optional upload demo; launch is commented out |

## Evaluation

Sketch preservation compares skeletonized white strokes in the control image with Canny edges in the generated image.

- **Edge recall** is the fraction of sketch-line pixels that have a generated edge within 5 pixels. Higher means the lines were kept.
- **Chamfer distance** is the mean distance, in pixels, from sketch-line pixels to the nearest generated edge. Lower means closer alignment.

These scores measure structural alignment. They do not measure prompt accuracy, photorealism, or whether a person would prefer the image. Read them together with the visual grids.

Saved tables:

- `outputs/metrics_baseline.csv`
- `outputs/metrics_controlnet_scale.csv`
- `outputs/experiment_runs.csv`
- `outputs/run_manifest.json`

## Data

Automatic sketches come from the Quick, Draw! simplified NDJSON files:

`https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson`

The notebook keeps a deterministic sample among the first 10,000 recognized drawings and records the source in `sketches/source_attribution.csv`. Quick, Draw! is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The `house` category is used as the building sketch.

If you add your own images, you are responsible for having the right to use them.

## Limitations

- Edge proximity is not a human preference score.
- Results depend on the prompt, preprocessing, seed, model revision, and GPU.
- A fixed seed improves reproducibility and still will not match bit-for-bit across different hardware or library versions.
- The safety checker is disabled in the pipeline so conditioning is not interrupted by the default NSFW filter. Review outputs before you share them.
- A Hugging Face `local_dir_use_symlinks` message is a dependency deprecation warning. If the next cell prints `Pipeline ready`, the download succeeded.

## Repository layout

```text
.
├── README.md
├── requirements.txt
└── Sketches to Life.ipynb
```

Generated images, metrics, and the results ZIP are created when you run the notebook. They are not stored in this repository.
