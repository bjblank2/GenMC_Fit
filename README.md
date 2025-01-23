# GenMC_Fit
## Overview
GenMC_Fit is a Python-based component of the GenMC-MA toolkit, designed to parameterize lattice models from density functional theory (DFT) datasets. It operates as part of the broader GenMC-MA workflow, which is used to model the thermodynamic, magnetic, and structural properties of alloys and compounds. This tool specifically takes DFT-derived datasets and calculates effective cluster interaction (ECI) terms or other model parameters through a regression approach. These parameters are then used to predict material properties and support Monte Carlo simulations.

---

## Key Features
- Parameterizes lattice models, including cluster expansions, Ising models, and Potts models.
- Supports multiple linear regression techniques: Ridge, Lasso, Least Squares, and Elastic Net.
- Handles symmetry-based clustering and decorations for complex lattice systems.
- Outputs parameterized models for use with GenMC_Run.

---

## Prerequisites
- **Python 3.7+**
- Required Python libraries: `numpy`, `pymatgen`, `json`, `yaml`
- DFT datasets with corresponding POSCAR, OUTCAR, and CONTCAR files
- An input parameter file (`param_in`)

---

## Installation
1. Clone the repository containing GenMC_Fit:
   ```bash
   git clone <repository_url>
   ```
2. Install required Python dependencies:
   ```bash
   pip install numpy pymatgen yaml
   ```

---

## Input Files
### 1. **`param_in` File**
This file defines configuration options for GenMC_Fit. Below is an example structure:

```
lat_in: 'POSCAR'  # Path to the lattice structure file
data_file: 'data_file.json'  # Path to compacted DFT dataset
clust_in: 'cluster_in.json'  # Path to cluster definition file
species: ['Ni', 'Mn', 'In']  # Atomic species in the system
fit_ridge: true  # Use Ridge regression for parameter fitting
fit_lasso: false  # Use LASSO regression
fit_eln: false  # Use Elastic Net regression
rescale_enrg: false  # Energy rescaling option
do_fit: true  # Perform regression fitting
do_count: false  # Count clusters (set to true if counts are not precomputed)
```

### 2. **Cluster Definition File (`cluster_in.json`)**
This file defines the motifs and types of clusters to be analyzed. An example format:

```json
{
  "List": [
    [ [0, 0, 0], [1], [0] ],
    [ [ [0, 0, 0], [2.6, 0.0, 0.0] ], [1], [1] ]
  ]
}
```

---

## Running GenMC_Fit
1. Prepare the required input files (`param_in`, `cluster_in.json`, etc.).
2. Run the script using the following command:

   ```bash
   python main.py
   ```

3. The script performs two main tasks depending on the `param_in` settings:
   - **Cluster Counting:** When `do_count: true`, cluster occurrences are computed for the input dataset.
   - **Model Fitting:** When `do_fit: true`, the regression models are applied to fit the lattice model parameters.

---

## Outputs
1. **`CLUSTERS` File**: Contains the parameterized lattice model.
2. **Fitted Coefficients**: Written to files like `eci_out` or `eci_out_lasso` depending on the regression type.
3. **Cluster Counts**: Written to `count_out` if `do_count` is enabled.

---

## Example Workflow
### Step 1: Cluster Counting
Update `param_in`:
```yaml
do_count: true
do_fit: false
```
Run the script:
```bash
python main.py
```
Output: `count_out` (contains computed cluster counts).

### Step 2: Parameter Fitting
Update `param_in`:
```yaml
do_count: false
do_fit: true
```
Run the script:
```bash
python main.py
```
Output: `eci_out` (contains fitted parameters).

---

## Tips for Model Building
1. **Symmetry Handling:** Ensure accurate symmetry analysis by verifying input POSCAR files.
2. **Energy Rescaling:** Use `rescale_enrg: true` when working with scaled energy values.
3. **Validation:** Perform cross-validation by adjusting regression parameters and methods.

---

## References
For more information on GenMC-MA, refer to the full documentation or contact the development team.

