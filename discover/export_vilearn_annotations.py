import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import discover_utils.data.handler.nova_db_handler as mh

"""
Export ViLearn annotations from NOVA/Discover into CSV files.

Only annotations are exported (streams are already written to disk by Discover).
"""

# ------------------------
# CONFIG
# ------------------------
DATASET = "vilearn_more"
SET_FILES = ["discover/vilearn_more.set"]
DOTENV_PATH = "discover/.env"
OUTPUT_ROOT = Path("data/discover")

# Annotation schemes to export and the annotator(s) to use for each scheme.
# task engagement: both annotators (group TE = 2-annotator mean downstream);
# the 60Hz scheme is the fallback for sessions annotated at 60 Hz.
SCHEME_ANNOTATOR = {
    "transcript": ["helenrisack"],
    "engagement": ["helenrisack"],
    "task engagement": ["helenrisack", "carlosgonzalez"],
    "task engagement60Hz": ["helenrisack", "carlosgonzalez"],
    "sentiment": ["carlosgonzalez"],
    "arousal": ["carlosgonzalez"],
    "dominance": ["carlosgonzalez"],
    "valence": ["carlosgonzalez"],
}

# Expected roles per scheme (avoid reporting expected missing data)
SCHEME_ROLES = {
    "transcript": ["p_blue", "p_green", "p_red"],
    "engagement": ["p_blue", "p_green", "p_red"],
    "task engagement": ["group"],
    "task engagement60Hz": ["group"],
    "sentiment": ["p_blue", "p_green", "p_red"],
    "arousal": ["group"],
    "dominance": ["group"],
    "valence": ["group"],
}

# ------------------------
# HELPERS
# ------------------------

def load_sessions() -> list[str]:
    sessions: list[str] = []
    for path in SET_FILES:
        entries = [line.strip() for line in Path(path).read_text().splitlines()]
        sessions.extend([e for e in entries if e])
    if not sessions:
        raise ValueError("No sessions found in SET_FILES. Each line must be a session name.")
    return sorted(sessions)

def expected_roles(session: str, scheme: str) -> list[str]:
    roles = SCHEME_ROLES.get(scheme, [])
    if "triad" in session.lower():
        return roles
    return [r for r in roles if r != "p_red"]


def safe_filename(name: str) -> str:
    return name.replace(" ", "_")


# ------------------------
# MAIN
# ------------------------

def main() -> None:
    load_dotenv(DOTENV_PATH)
    dbconf = {
        "db_host": os.environ["DBHOST"],
        "db_port": int(os.environ["DBPORT"]),
        "db_user": os.environ["DBUSER"],
        "db_password": os.environ["DBPASSWORD"],
    }

    ah = mh.AnnotationHandler(**dbconf)
    sessions = load_sessions()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Exporting annotations for {len(sessions)} sessions...")
    missing = []
    for ses in sessions:
        ses_dir = OUTPUT_ROOT / ses
        ses_dir.mkdir(parents=True, exist_ok=True)

        for scheme, annotators in SCHEME_ANNOTATOR.items():
            for annotator in annotators:
                for role in expected_roles(ses, scheme):
                    try:
                        anno = ah.load(dataset=DATASET, session=ses, scheme=scheme, role=role, annotator=annotator)
                    except Exception:
                        missing.append((ses, scheme, role, annotator))
                        continue

                    if getattr(anno, "data", None) is None:
                        continue

                    df = pd.DataFrame(anno.data)
                    if df.empty:
                        continue

                    filename = f"{safe_filename(scheme)}.{role}.{annotator}.csv"
                    df.to_csv(ses_dir / filename, index=False)

    if missing:
        print("Missing annotations (unexpected):")
        for ses, scheme, role, annotator in missing:
            print(f"  {ses} | {scheme} | {role} | {annotator}")

    print("Done")


if __name__ == "__main__":
    main()
