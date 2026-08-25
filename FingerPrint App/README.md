# Student Attendance Fingerprint System

A local Streamlit prototype for fingerprint-based classroom attendance. Its
production enhancement engine follows the binarization-based filtering method
described by Greenberg, Aladjem, Kogan and Dimitrov, while fingerprint identity
matching remains a separate downstream operation.

## Run the application

Install the packages once:

```powershell
python -m pip install -r requirements.txt
```

Then start the system with the requested command:

```powershell
python -m streamlit run app.py
```

If `python` is not available in a new PowerShell window, activate the included
environment first with `.\.venv\Scripts\Activate.ps1`. Streamlit normally opens
`http://localhost:8501` automatically.

## Implemented filtering technique

The same five filtering stages are executed for every capture:

1. Local histogram equalization using an `11 x 11` neighbourhood.
2. Pixel-wise adaptive Wiener noise filtering using a `3 x 3` neighbourhood.
3. Adaptive binarization against the `13 x 13` local intensity mean.
4. Morphological thinning of black ridges to one-pixel centre lines.
5. Binary ridge post-processing that removes connected false ridges shorter
   than 10 pixels and closes small gaps.

A variance-based foreground guard excludes blank background and scanner frames
before the five stages. This is clearly labelled as a system extension rather
than a stage claimed by the paper. Resizing and padding only create a consistent
array for local database comparison.

The application no longer presents STFT, orientation estimation, ridge-frequency
calculation or Gabor filtering as its production enhancement path. Any values in
the earlier Assignment IP benchmark are historical report values and must not be
described as newly measured performance.

## Attendance identification flow

After enhancement, the separate attendance function:

1. Compares a new scan with each canonical greyscale enrolment reference using
   SIFT keypoints and RANSAC geometric verification.
2. Supports that primary score with enhanced-image ORB, aligned structural and
   spectral evidence.
3. Requires both the configured similarity threshold and a clear lead over the
   runner-up before recording attendance.
4. Records a student only once for the selected class session.

Keeping the enrolment upload as a canonical reference is what permits a different
capture of the same finger to match; the system does not require identical image
pixels. The transparent score is suitable for a classroom prototype, not a
forensic or high-security biometric certification.

## Included application features

- Student enrolment with one to three captures of the same finger
- One-to-many local fingerprint identification
- Capture-quality, match-threshold and ambiguity guards
- Present/late classification and duplicate prevention
- SQLite persistence for students, templates, sessions and attendance
- Dashboard, class roster and audit information
- CSV and polished PDF attendance-register exports
- Labelled synthetic demo cohort for end-to-end demonstrations
- Algorithm Studio with all five live intermediate filtering results

## Project modules

| File | Responsibility |
|---|---|
| `app.py` | Streamlit navigation and application workflows |
| `fingerprint_processing.py` | Paper-aligned filtering, quality measurements and comparison helpers |
| `matching.py` | One-to-many identification and attendance-status logic |
| `database.py` | SQLite schema and persistence operations |
| `reporting.py` | PDF attendance-register generator |
| `ui.py` | Shared dashboard styling and interface components |
| `tests/test_system.py` | Processing, matching, database and report regression tests |
| `FILTERING_METHODOLOGY.md` | Assignment-ready filtering rationale, pseudocode and evaluation guidance |

## Demonstration workflow

1. Open **System & help** and select **Load or repair demo cohort**.
2. Open **Mark attendance** and select the generated BMDS2133 session.
3. Choose **Use labelled demo scan**, select a student and run verification.
4. Review the matching evidence and filtering-stage images.
5. Open **Attendance records** to export the class register.

Synthetic demo records are marked `(Demo)` and must not be reported as real
experimental measurements.

## Data and privacy

Runtime data is stored under `data/` and excluded from Git by default. New
enrolments retain a consistently sized greyscale reference under
`data/references/` and a processed binary template under `data/templates/`.
Existing user records are migrated without deletion and older templates continue
through the enhanced-image fallback.

Fingerprints are sensitive personal data. Real deployment requires informed
consent, access control, encryption at rest and a documented retention policy.

## Source

Greenberg, S., Aladjem, M., Kogan, D., & Dimitrov, I. (2000). *Fingerprint image
enhancement using filtering techniques*. Proceedings of the 15th International
Conference on Pattern Recognition. The extended journal article is available at
https://doi.org/10.1006/rtim.2001.0283.

AI assistance used to produce or refine a submission should be declared in the
assignment's AI Usage Disclosure Form.
