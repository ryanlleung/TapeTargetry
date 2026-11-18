# TapeTargetry

Jupyter notebooks for estimating the coating thickness of targets on a Kapton tape coated by plasma sputtering or e-beam deposition.

There are 180 target points on the tape: 96 on the front side and 84 on the back. Their (x, y) positions on a 237.62 × 237.62 mm baseplate are known. Each notebook does the same basic job:

1. Fit a **radial thickness profile** (thickness factor vs. distance from the baseplate centre, 0–160 mm) to measured data.
2. Evaluate that profile at each target's radius.
3. Report how many targets fall within a tolerance band (thickness factor 0.85–1.15), and export the per-target thickness factors to CSV.

A thickness factor is the local thickness divided by the average thickness ("Percentage of average" in the source spreadsheets).

The sibling repository **CoatingAnalysis** models coating thickness profiles in more depth.

## Repository layout

```
.
├── tapeman.ipynb          # Plasma sputterer: single-profile fit -> target thickness factors
├── superposer.ipynb       # Plasma sputterer: optimal blend of two sputter-head profiles
├── ebeamer.ipynb          # E-beam: optimal blend of two measured profiles (1/3 and 2/3 full crucible)
├── tapenn.ipynb           # E-beam: neural-network model of thickness vs (radius, crucible fullness)
├── plotqt.ipynb           # Prototype PyQt5 GUI ("Thickness Profile Plotter")
├── front_data.csv         # Target coordinates, front side of tape (input)
├── back_data.csv          # Target coordinates, back side of tape (input)
├── thickness_data.csv     # Sputterer radial thickness profile
├── thickness_data.xlsx    # Same profile as an Excel sheet (read by old/profilemap.ipynb)
├── target_data*.csv       # Per-target thickness factors (outputs)
├── ebeam-33%.csv, ebeam-66%.csv  # Per-target thickness factors from tapenn (outputs)
├── ebeam/
│   ├── EB1321 Coating analysis.xlsx  # E-beam run EB1321, crucible 1/3 full
│   ├── EB1322 Coating analysis.xlsx  # E-beam run EB1322, crucible 2/3 full
│   └── models/model-YYMMDD-HHMM.h5   # Keras models saved by tapenn.ipynb
├── fit/
│   ├── fitter.ipynb       # Experiments: placing 16 glass slides in an annulus without overlap
│   └── tf_test.ipynb      # TensorFlow GradientTape scratch tests
└── old/
    ├── profilemap.ipynb   # Earlier version of the profile fit that loads thickness_data.xlsx
    └── vidwrite.ipynb     # Turns frames/frame1..180.png into vid3.mp4 with OpenCV
```

## Notebooks

All notebooks use relative paths, so run them from the repository root.

### `tapeman.ipynb`: plasma sputterer, single profile
- **Input:** the radial profile, hard-coded in the notebook. The values are the same as `thickness_data.csv`. The notebook also reads `front_data.csv` and `back_data.csv`.
- **Method:** fits a sum of two logistic functions with `scipy.optimize.curve_fit`, then smooths the result with a clamped `CubicSpline` so the slope is zero at the centre. Plots the 3D thickness map, target locations, thickness factor per target, and violin/box plots.
- **Outputs:**
  - `target_data.csv`: all 180 targets.
  - `target_data_capped.csv`: only the targets within 0.85–1.15.
- **Also included:** a "Slides Coating" section that masks the thickness map onto 16 glass slides (75 × 25 mm) laid out on a 240 mm plate.

### `superposer.ipynb`: plasma sputterer, two heads
- **Input:** two hard-coded radial profiles, labelled "Angled Head" and "Vertical Head". The notebook also reads `front_data.csv` and `back_data.csv`.
- **Method:** fits each profile with the double-logistic function. It then finds the mixing ratio `r` for `r·f1 + (1−r)·f2` with Nelder–Mead (`scipy.optimize.minimize`) in two ways:
  - (a) minimising the MSE from 1 across the whole plane;
  - (b) minimising the standard deviation of the thickness factor over the 180 target radii.

  It plots KDE distributions of the thickness factor against the ratio.
- **Output:** `target_data_opt.csv`, using the ratio from (b).

### `ebeamer.ipynb`: e-beam, two profiles
- The same workflow as `superposer.ipynb`, applied to the e-beam profiles from `ebeam/EB1321` (1/3 full) and `ebeam/EB1322` (2/3 full). The values are hard-coded as fractions of "Percentage of average".
- Its plot labels still say "Angled Head" / "Vertical Head", carried over from `superposer.ipynb`.
- **Output:** `target_data_opt.csv`. This is the same file name `superposer.ipynb` writes, so whichever notebook runs last overwrites the other's output.

### `tapenn.ipynb`: e-beam, neural network
- **Input:** the same two e-beam profiles, with the two training points per radius set to crucible fullness 0.333 and 0.667. The notebook also reads `front_data.csv` and `back_data.csv`.
- **Method:** trains a Keras MLP (Dense 128 → 64 → 1, GELU activation, Adam optimiser, MSE loss, 30,000 epochs). The input is (normalised radius, fullness) and the output is the thickness factor. It saves the model to `ebeam/models/model-<YYMMDD-HHMM>.h5`, or loads an existing one (by default `model-230628-1604.h5`). It then predicts the thickness map and the per-target thickness factors for a chosen `full` value.
- **Output:** `ebeam-66%.csv` as written. `ebeam-33%.csv` was presumably produced the same way, with `full` set to about 0.33.

### `plotqt.ipynb`: GUI prototype
A work-in-progress PyQt5 window titled "Thickness Profile Plotter". It has an editable radius/thickness-factor table and embedded 2D and 3D Matplotlib canvases. For now those canvases show placeholder sine plots. The notebook also contains a clipboard "Paste Table" demo.

## Data files

| File | Format | Contents |
|---|---|---|
| `front_data.csv`, `back_data.csv` | No header, 2 columns | Target (x, y) in mm from the drawing (y is negated on load). Front: 96 rows. Back: 84 rows. |
| `thickness_data.csv` / `.xlsx` | `Distance from centre, Percentage of Average` | Sputterer radial profile, 0–160 mm in 10 mm steps (17 points). |
| `ebeam/EB13xx Coating analysis.xlsx` | Spreadsheet | Distance from the baseplate centre (mm), thickness (nm), difference from the mean, and percentage of average. The notes cite a crystal measurement of 199 nm on the e-beam plant. |
| `target_data.csv`, `target_data_capped.csv`, `target_data_opt.csv`, `ebeam-33%.csv`, `ebeam-66%.csv` | `Target Number, Thickness Factor` | Per-target results (180 rows; the capped file has 146). |

Target numbering is "unravelled" along the tape. Targets alternate between the front and back in blocks of 12: 1–12 are on the front, 13–24 on the back, and so on.

## Dependencies

These are taken from the notebook imports:

- `numpy`, `pandas`, `matplotlib`, `scipy`: all notebooks
- `openpyxl`: needed by `pandas.read_excel` for the `.xlsx` files
- `tensorflow` (2.10): `tapenn.ipynb`, `fit/`
- `PyQt5`: `plotqt.ipynb`
- `shapely`: `fit/fitter.ipynb`
- `opencv-python` (`cv2`): `old/vidwrite.ipynb`
- `jupyter`

## Installation

Ideally, create a new environment with conda. The notebooks were run with Python 3.9.

```
conda create -n <NAME> python=3.9
conda activate <NAME>
```

Don't use conda to install TensorFlow. Use pip instead:

```
pip install tensorflow==2.10
```

Then install the remaining packages:

```
pip install numpy pandas matplotlib scipy openpyxl jupyter pyqt5 shapely opencv-python
```

## Usage

```
jupyter notebook
```

Open a notebook from the repository root and run all cells. The tolerance band is set with `TF_FLOOR` and `TF_CEILING`. In `tapenn.ipynb`, set `model_path` to a model in `ebeam/models/` (to skip training) and `full` to the crucible fullness (0–1) before running the prediction cells.

## License

Apache License 2.0. See [LICENSE](LICENSE).
