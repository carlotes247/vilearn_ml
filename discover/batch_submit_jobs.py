import json
import os
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv

"""
Batch submit extraction jobs for ViLearn in DISCOVER.

Configure the variables in the CONFIG section before running.
"""

# ------------------------
# CONFIG
# ------------------------
DATASET = "vilearn_more"
DATASET_PATH = f"/mnt/datasets/nova/data/{DATASET}"
CG = "carlosgonzalez"
HR = "helenrisack"
SESSION_CHUNK_SIZE = 5  # used for memory-heavy jobs (e.g., libreface)

# Set files are the source of truth. Each line must be a session name.
# Set files are also choosable from NOVA in a dropdown.
SET_FILES = [
    "discover/vilearn_more.set",
]

# Roles observed in sessions: group, p_blue, p_green, p_red
# Participant roles differ by session type (dyad vs triad).
PARTICIPANT_ROLES_BASE = ["p_blue", "p_green"]
PARTICIPANT_ROLE_TRIAD = "p_red"

# .env file providing DBHOST, DBPORT, DBUSER, DBPASSWORD
DOTENV_PATH = "discover/.env"  # copy .env.example to .env and edit with correct values

URL = "https://localhost:27014/process"  # enter DISCOVER URL for job submission (e.g., https://<ip>:<port>/process)
HEADERS = {"Content-type": "application/x-www-form-urlencoded"}
VERIFY_SSL = False  # set True if you have a trusted cert

# ------------------------
# SESSION DISCOVERY
# ------------------------

def load_sessions() -> list[str]:
    if not SET_FILES:
        raise ValueError("SET_FILES is empty. Provide at least one .set file.")

    sessions: list[str] = []
    for path in SET_FILES:
        entries = [line.strip() for line in Path(path).read_text().splitlines()]
        sessions.extend([e for e in entries if e])

    if not sessions:
        raise ValueError("No sessions found in SET_FILES. Each line must be a session name.")

    return sorted(sessions)


# ------------------------
# JOB DEFINITIONS
# ------------------------

# Each job template can be expanded per role using {role}
# "roles" controls which roles the job should be submitted for.
JOBS: list[dict] = []

# opensmile (group audio)
JOBS.append({
    "roles": ["group"],
    "trainerFilePath": "opensmile/opensmile.trainer",
    "leftContext": "120",
    "rightContext": "120",
    "frameSize": "40",
    "data": (
        '[{"id":"input_audio","type":"input","src":"db:stream:Audio","name":"audio","role":"{role}","active":true},'
        '{"id":"output_stream","type":"output","kind":"SSIStream:feature","src":"db:stream:SSIStream:feature","name":"opensmile","role":"{role}","active":true}]'
    ),
    "options": '{"feature_set":"eGeMAPSv02","feature_lvl":"Functionals","file_num_workers":0}',
    "force": "False",
})
'''
# sentiment (participant transcripts)
JOBS.append({
    "roles": ["participant"],
    "trainerFilePath": "sentiment\\sentiment.trainer",
    "leftContext": "0",
    "rightContext": "0",
    "frameSize": "40",
    "data": (
        '[{"id":"transcript","type":"input","src":"db:annotation","scheme":"transcript","annotator":"{HR}","role":"{role}","active":true},'
        '{"id":"sentiment","type":"output","src":"db:annotation","scheme":"sentiment","annotator":"{CG}","role":"{role}","active":true},'
        '{"id":"embedding","type":"output","kind":"SSIStream","src":"db:stream:SSIStream","name":"sentiment","role":"{role}","active":true}]'
    ),
    "options": '{"model_path":"cardiffnlp/twitter-xlm-roberta-base-sentiment"}',
    "force": "False",
})

# emow2v (group audio)
JOBS.append({
    "roles": ["group"],
    "trainerFilePath": "emow2v/emow2v.trainer",
    "leftContext": "0",
    "rightContext": "0",
    "frameSize": "40",
    "data": (
        '[{"id":"audio","type":"input","src":"db:stream:Audio","name":"audio","role":"{role}","active":true},'
        '{"id":"arousal","type":"output","src":"db:annotation","scheme":"arousal","annotator":"{CG}","role":"{role}","active":true},'
        '{"id":"dominance","type":"output","src":"db:annotation","scheme":"dominance","annotator":"{CG}","role":"{role}","active":true},'
        '{"id":"valence","type":"output","src":"db:annotation","scheme":"valence","annotator":"{CG}","role":"{role}","active":true},'
        '{"id":"embedding","type":"output","kind":"SSIStream","src":"db:stream:SSIStream","name":"emow2v","role":"{role}","active":true}]'
    ),
    "options": '{"batch_size":250}',
    "force": "False",
})
'''
# ------------------------
# SUBMIT
# ------------------------

def main() -> None:
    if not VERIFY_SSL:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    load_dotenv(DOTENV_PATH)
    dbconf = {
        "dbHost": os.environ["DBHOST"],
        "dbPort": os.environ["DBPORT"],
        "dbUser": os.environ["DBUSER"],
        "dbPassword": os.environ["DBPASSWORD"],
    }

    sessions = load_sessions()
    triad_sessions = [s for s in sessions if "triad" in s.lower()]

    def render_data(template: str, role: str) -> str:
        return (
            template
            .replace("{role}", role)
            .replace("{CG}", CG)
            .replace("{HR}", HR)
        )

    job_queue: list[dict] = []
    for job in JOBS:
        job_roles = job["roles"]
        resolved_roles: list[str] = []
        for role in job_roles:
            if role == "participant":
                resolved_roles.extend(PARTICIPANT_ROLES_BASE)
                if triad_sessions:
                    resolved_roles.append(PARTICIPANT_ROLE_TRIAD)
            else:
                resolved_roles.append(role)

        for role in resolved_roles:
            target_sessions = triad_sessions if role == PARTICIPANT_ROLE_TRIAD else sessions
            if not target_sessions:
                continue

            job_copy = {k: v for k, v in job.items() if k != "roles"}
            job_copy["data"] = render_data(job_copy["data"], role)
            job_copy["sessions_list"] = target_sessions
            job_queue.append(job_copy)

    print(f"Submitting {len(job_queue)} jobs for {len(sessions)} sessions...")
    for i, j in enumerate(job_queue):
        j |= dbconf
        trainer_path = j["trainerFilePath"].replace("\\", "/")
        trainer_name = Path(trainer_path).stem
        role_name = j["data"].split('"role":"', 1)[1].split('"', 1)[0]
        job_id_base = f"vl_{trainer_name}_{role_name}"
        if "libreface" in j["trainerFilePath"]:
            target_sessions = j.pop("sessions_list")
            chunks = [target_sessions[k:k + SESSION_CHUNK_SIZE] for k in range(0, len(target_sessions), SESSION_CHUNK_SIZE)]
            for k, ses in enumerate(chunks):
                j |= {"jobID": f"{job_id_base}_{k:02}", "sessions": json.dumps(ses), "dataset": DATASET}
                req = requests.post(URL, headers=HEADERS, data=j, verify=VERIFY_SSL)
                if req.status_code != 200:
                    print(f"Canceled at job {j['jobID']} (HTTP {req.status_code})")
                    return
        else:
            target_sessions = j.pop("sessions_list")
            j |= {"jobID": job_id_base, "sessions": json.dumps(target_sessions), "dataset": DATASET}
            req = requests.post(URL, headers=HEADERS, data=j, verify=VERIFY_SSL)
            if req.status_code != 200:
                print(f"Canceled at job {j['jobID']} (HTTP {req.status_code})")
                return

    print("Done")


if __name__ == "__main__":
    main()
