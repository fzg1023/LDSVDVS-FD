# Data and code sources

The arrays in the separate asset folder are copies of the windowed data used for the paper. They are not collected by the authors of this bundle. This bundle does not assign a new license to any source data. Public access to a download is not itself a grant to redistribute it. Verify the current terms or obtain permission before uploading prepared arrays, noisy waveform copies, prediction arrays, or fitted observers containing training features. Publish the preparation code, protocols and source citations even if the arrays cannot be redistributed.

| Dataset | Official source / citation | Treatment in this bundle |
|---|---|---|
| HUST | [Dataset repository](https://github.com/CHAOZHAO-1/HUSTbearing-dataset), [Zhao et al.](https://doi.org/10.1016/j.ress.2024.109964) | Windows from nine records at the 20 Hz motor setting. |
| BJTU-RAO | [Qin et al., dataset tutorial](https://doi.org/10.1109/ACCESS.2025.3551603) | Six selected gearbox states at one condition. The protocol lists the exact filenames. |
| HIT | [Dataset repository](https://github.com/HouLeiHIT/HIT-dataset), [Hou et al.](https://doi.org/10.37965/jdmd.2023.314) | Casing acceleration at measurement point 4; 3,000/3,600 rpm; four states. |
| Northeastern University gearbox | Recording IDs and source metadata in `protocols/northeastern_university/`. | Supplement only. Archive a citable download location and check redistribution permission before public release. |
| Ottawa | Recording IDs and source metadata in `protocols/ottawa/`. | Supplement only. Verify terms for the exact nine selected records. |

Code attributions: [Xformer](https://github.com/gladzhang/Xformer) (upstream README declares Apache 2.0); this bundle includes the paper's own 1-D adaptation, not the original 2-D package or weights. [ARDiff](https://doi.org/10.1016/j.ymssp.2025.113726) is the basis for the paper's independent RMS-adapted implementation. Patch Mamba depends on the external `mambapy` package, which is not vendored. Preserve any existing source notices and include the applicable third-party license text for any upstream code that is redistributed. The authors should choose a license for their own code and checkpoints before publication. Model weights are provided locally; use a release asset or data repository instead of adding them directly to Git.
