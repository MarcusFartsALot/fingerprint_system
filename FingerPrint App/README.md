# Student Attendance Fingerprint System

A local Streamlit prototype for fingerprint-based classroom attendance. Its
production enhancement engine uses CLAHE, edge-preserving bilateral filtering
and mild unsharp enhancement, while identity matching remains separate.

## Run the application

### Easy start and stop

Double-click `START_APP.bat` to start the server and open the app. Double-click
`STOP_APP.bat` when finished. From PowerShell, the same shortcuts are:

```powershell
.\START_APP.bat
.\STOP_APP.bat
```

The server runs quietly in the background. Diagnostic output is available in
`data/streamlit-output.log` and `data/streamlit-error.log` if needed. Running
`START_APP.bat` again detects the existing managed server instead of creating a
duplicate process. `STOP_APP.bat` stops the Python listener on this project's
reserved port even if the app was originally started manually. Closing the
browser tab is optional and does not stop the server.

The included Streamlit configuration binds the app to `127.0.0.1`, so it is
available only on the current computer by default. `.gitignore` excludes the
SQLite database, student information, fingerprint references/templates and
runtime logs; do not force-add those sensitive files to GitHub.

### Manual method

Install the packages once:

```powershell
python -m pip install -r requirements.txt
```

Then start the system with the requested command:

```powershell
python -m streamlit run app.py
```

When started manually, stop it in the same terminal by pressing `Ctrl+C` once.

If `python` is not available in a new PowerShell window, activate the included
environment first with `.\.venv\Scripts\Activate.ps1`. Streamlit normally opens
`http://localhost:8501` automatically.

### Run on macOS

After cloning the project on a Mac, create a local environment and start it with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Stop it with `Control+C`. Runtime databases, biometric images, logs and backups
remain ignored by Git and are created locally on each computer.

After the one-time installation above, Finder users can double-click
`START_APP.command` to start the background server and open the browser. Double-
click `STOP_APP.command` to stop it. The Git branch stores both files as
executable. If macOS blocks the first launch, Control-click the file, choose
**Open**, and approve it. If executable permission was lost while copying the
folder outside Git, restore it once with:

```bash
chmod +x START_APP.command STOP_APP.command
```

For a sharper live capture, a compatible iPhone can appear to macOS as a webcam
through Continuity Camera. It works wirelessly or over USB. Connect and trust the
Mac, enable Continuity Camera under **iPhone Settings > General > AirPlay &
Continuity**, then select the iPhone in the browser's camera settings before
opening **Live camera capture**. The app receives the resulting camera frame in
the same format as any other webcam; no fingerprint-processing code change is
required.

## Implemented filtering technique

Foreground extraction is always executed first, followed by the same six
filtering stages for every capture:

0. Centre-colour, boundary and ridge-coherence segmentation crops and normalizes
   the fingerprint pad while whitening its background.

1. CLAHE local contrast enhancement using an `8 x 8` tile grid and clip limit `2.0`.
2. Edge-preserving bilateral filtering using a `5 x 5` neighbourhood.
3. Mild unsharp enhancement that reinforces ridge edges already present.
4. Adaptive binarization against the `13 x 13` local intensity mean.
5. Morphological thinning of black ridges to one-pixel centre lines.
6. Binary ridge post-processing that removes connected false ridges shorter
   than 10 pixels and closes small gaps.

A centre-seeded GrabCut mask combines colour and object boundaries to isolate the
fingertip from a phone-photo background. The detected pad is cropped and scaled
consistently before filtering and matching.

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
- Browser camera capture with automatic post-capture attendance verification
- Capture-quality, match-threshold and ambiguity guards
- Present/late classification and duplicate prevention
- SQLite persistence for students, templates, sessions and attendance
- Dashboard, class roster and audit information
- CSV and polished PDF attendance-register exports
- Labelled synthetic demo cohort for end-to-end demonstrations
- Algorithm Studio with all six live intermediate filtering results

## Project modules

| File | Responsibility |
|---|---|
| `app.py` | Streamlit navigation and application workflows |
| `fingerprint_processing.py` | Ridge-preserving enhancement, quality measurements and comparison helpers |
| `matching.py` | One-to-many identification and attendance-status logic |
| `database.py` | SQLite schema and persistence operations |
| `reporting.py` | PDF attendance-register generator |
| `ui.py` | Shared dashboard styling and interface components |
| `tests/test_system.py` | Processing, matching, database and report regression tests |
| `FILTERING_METHODOLOGY.md` | Assignment-ready filtering rationale, pseudocode and evaluation guidance |

### Folder guide

| Folder | Keep? | Purpose |
|---|---|---|
| `.streamlit/` | Yes | Local Streamlit server configuration |
| `data/` | Yes | Local database, enrolled fingerprint references, reports and runtime logs |
| `scripts/` | Yes | Start/stop logic used by the double-click launcher files |
| `tests/` | Recommended | Automated evidence that processing, matching, duplicate prevention and database editing still work |
| `__pycache__/` | No | Generated Python cache; safe to remove and already ignored by Git |

For a presentation, open the application using `START_APP.command` on macOS or
`START_APP.bat` on Windows. Present the interface, then use
`FILTERING_METHODOLOGY.md` when explaining the enhancement pipeline. The source
and test folders do not need to be opened during the live demonstration.

## Demonstration workflow

1. Open **Student enrolment**, enter a student and upload two or three captures
   of the same finger.
2. Open **Class sessions** and create a class session.
3. Open **Mark attendance**, select that session and use either **Upload
   fingerprint photo** or **Live camera capture**.
4. Review the matched enrolment reference, match score and technical filter
   evidence.
5. Open **Attendance records**, select the session and export its register.

For a physical capture, choose **Live camera capture**, allow browser camera
permission and take a still image. The new capture is processed and matched
automatically. This is convenient acquisition, not continuous-video liveness or
anti-spoofing.

## Data and privacy

Runtime data is stored under `data/` and excluded from Git by default. New
enrolments retain a consistently sized greyscale reference under
`data/references/` and a processed binary template under `data/templates/`.
Existing user records are migrated without deletion and older templates continue
through the enhanced-image fallback.

Fingerprints are sensitive personal data. Real deployment requires informed
consent, access control, encryption at rest and a documented retention policy.

## Sources

The implementation uses the documented OpenCV CLAHE and bilateral-filtering
operations. See `FILTERING_METHODOLOGY.md` for the rationale, parameters,
limitations and evaluation design. STFT remains documented there as a rejected
experiment for these phone captures because it generated unsupported coarse
ridge patterns.

AI assistance used to produce or refine a submission should be declared in the
assignment's AI Usage Disclosure Form.
