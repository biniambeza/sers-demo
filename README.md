# HCC Triage â€” Serum SERS (research demo)

Scores serum SERS spectra for hepatocellular carcinoma (liver cancer) triage:
REFER / CLEAR / INCONCLUSIVE. Trained on 144 real public spectra
(Zenodo 4277797, 72 HCC + 72 healthy). Not a medical device. Not for clinical use.

Test files: `data/hcc_serum_sers/spectra/` (CTR = healthy, H0T = HCC).

Run locally: `pip install -r requirements.txt` then `streamlit run demo_app.py`.