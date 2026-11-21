# SERS Liver Cancer Triage (research demo)

Hybrid molecule + pattern model for hepatocellular carcinoma triage from serum
SERS spectra: REFER / CLEAR / INCONCLUSIVE. Trained on 144 real public spectra
(Zenodo 4277797, 72 HCC + 72 healthy). Not a medical device. Not for clinical use.

Demo files from unseen test patients: `demo_samples/` (answers in `answers.csv`).
Full dataset: `data/hcc_serum_sers/spectra/` (CTR = healthy, H0T = HCC).

Run locally: `pip install -r requirements.txt` then `streamlit run demo_app.py`.