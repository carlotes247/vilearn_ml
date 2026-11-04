import os
import re
import pandas as pd
import numpy as np

# Blink conventions (same rules as GroupCSVDataLoader.extract_valid_blinks_frames,
# after Nakano and Miyazaki, 2019):
#   * a frame counts as a blink frame when the openness value of BOTH eyes is
#     below `blink_openness_threshold`,
#   * a contiguous run of blink frames is one blink candidate,
#   * a candidate is only valid when its duration (first to last blink frame)
#     is between `min_blink_ms` and `max_blink_ms`,
#   * the onset of a valid blink is the timestamp of its FIRST blink frame.

def read_path_prefix() -> str:
    """
    Reads data/_path_prefix.txt and returns the first absolute, non-comment
    line. Used to resolve the relative recording paths stored in the dyad
    text files (e.g. 'Recordings/...').
    """
    # <root>/data_reading/features/this file -> root is two levels up
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    prefix_file = os.path.join(root, "data", "_path_prefix.txt")
    if not os.path.exists(prefix_file):
        return ""
    with open(prefix_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line or line.startswith("#")):
                continue
            if (line.startswith("/") or re.match(r"^[A-Za-z]:[\\/].*$", line)):
                return line.replace("/", os.sep)
    return ""

class VrcftEyeDataLoader:
    """
    Loads a per-participant eyedata CSV (VRCFT_EyeData.csv) from the
    2026 retreat dataset and extracts valid blinks (onset + duration).

    Expected format (one file per participant, comma separated, ~87Hz):
        NtpTimestampHex, TimestampUtc, NtpOffsetMs,
        LeftEyeOpenness, RightEyeOpenness,
        LeftGazeX, LeftEyeY, RightGazeX, RightGazeY,
        AutoBlinkOpenness, IsAutoBlinkActive

    Loaded dataframes expose both time references:
        * 'ts'     : UTC timestamp derived from 'TimestampUtc'
        * 'ts_ntp' : NTP timestamp = 'ts' + 'NtpOffsetMs' (ms)
    """

    required_columns = ["TimestampUtc", "LeftEyeOpenness", "RightEyeOpenness"]
    path_prefix = read_path_prefix()

    #region variables
    file_path: str
    participant_name: str
    data_loaded: bool
    raw_data: pd.DataFrame    # contains both 'ts' (UTC) and 'ts_ntp' (NTP) columns
    timestamps: list          # all UTC sample timestamps (pd.Timestamp, sorted)
    session_duration_seconds: float
    blink_onsets: list        # valid blink onsets (pd.Timestamp, sorted)
    blink_durations_ms: list  # valid blink durations in ms
    #endregion

    def __init__(self, file_path: str, participant_name: str = "P1"):
        self.file_path = file_path
        self.participant_name = participant_name
        self.data_loaded = False
        self.raw_data = None
        self.timestamps = []
        self.session_duration_seconds = 0.0
        self.blink_onsets = []
        self.blink_durations_ms = []

    @staticmethod
    def project_root() -> str:
        # <root>/data_reading/features/this file -> root is two levels up
        return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    #region load
    def load_data(self, sep: str = ",") -> bool:
        rel = self.file_path.replace("\\", "/").strip()
        candidates = []
        if self.path_prefix:
            candidates.append((self.path_prefix + rel).replace("/", os.sep))
        # fall back to the project root and the working directory
        candidates.append(os.path.join(self.project_root(), rel).replace("/", os.sep))
        candidates.append(os.path.join(os.getcwd(), rel).replace("/", os.sep))
        path = candidates[0]
        if not os.path.exists(path):
            for cand in candidates[1:]:
                if os.path.exists(cand):
                    path = cand
                    break
        try:
            df = pd.read_csv(path, sep=sep)
        except FileNotFoundError:
            print(f"Error: eyedata file not found: {path}")
            return False
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            print(f"Error: file {os.path.basename(path)} is missing columns {missing}; expected {self.required_columns}")
            return False
        # UTC reference (parsed TimestampUtc), kept as the 'ts' column.
        ts = pd.to_datetime(df["TimestampUtc"], utc=True).dt.tz_localize(None)
        # NTP reference: UTC corrected by the per-sample NtpOffsetMs (ms) offset,
        # kept as the 'ts_ntp' column. Both time references are retained.
        if "NtpOffsetMs" in df.columns:
            offset_ms = pd.to_numeric(df["NtpOffsetMs"], errors="coerce").fillna(0.0)
            ts_ntp = ts + pd.to_timedelta(offset_ms, unit="ms")
        else:
            ts_ntp = ts.copy()
        df = df.assign(ts=ts, ts_ntp=ts_ntp).sort_values("ts").reset_index(drop=True)
        self.raw_data = df
        self.timestamps = df["ts"].tolist()
        # interaction duration (used as the "session length" for per-minute blink rates)
        self.session_duration_seconds = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        self.data_loaded = True
        return True
    #endregion

    #region blink detection
    def detect_blinks(self,
                      blink_openness_threshold: float = 0.5,
                      min_blink_ms: float = 100.0,
                      max_blink_ms: float = 500.0) -> tuple[list, list]:
        """
        Detects valid blinks from the raw left/right eye openness values and
        fills blink_onsets (pd.Timestamp list) and blink_durations_ms.
        Returns (blink_onsets, blink_durations_ms).
        """
        self.blink_onsets = []
        self.blink_durations_ms = []
        if not self.data_loaded:
            print(f"Error: eyedata was not loaded for {self.participant_name} ({self.file_path}); cannot detect blinks.")
            return self.blink_onsets, self.blink_durations_ms
        df = self.raw_data
        if df is None or len(df) == 0:
            return self.blink_onsets, self.blink_durations_ms

        openness_below = ((df["LeftEyeOpenness"] < blink_openness_threshold) &
                          (df["RightEyeOpenness"] < blink_openness_threshold)).to_numpy()
        # find contiguous runs of blink frames
        padded = np.concatenate(([False], openness_below, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) == 0:
            print(f"{self.participant_name}: no blink frames detected (openness threshold {blink_openness_threshold}).")
            return self.blink_onsets, self.blink_durations_ms

        ts = df["ts"].to_numpy()
        for s, e in zip(starts, ends):
            onset = ts[s]
            offset = ts[e]
            duration_ms = (pd.Timestamp(offset) - pd.Timestamp(onset)).total_seconds() * 1000.0
            if min_blink_ms <= duration_ms <= max_blink_ms:
                self.blink_onsets.append(pd.Timestamp(onset))
                self.blink_durations_ms.append(duration_ms)

        # onsets are already in temporal order because runs were found in order
        self.blink_onsets.sort()
        return self.blink_onsets, self.blink_durations_ms
    #endregion

    #region getters
    def get_blink_onsets(self) -> list:
        return self.blink_onsets

    def get_blink_durations_ms(self) -> list:
        return self.blink_durations_ms

    def mean_blink_duration_ms(self) -> float:
        if not self.blink_durations_ms:
            return float("nan")
        return float(np.mean(self.blink_durations_ms))
    #endregion

    #region export
    def export_full_csv(self, out_path: str = None) -> str:
        """
        Exports the full loaded dataframe (all original CSV columns plus the
        derived 'ts' column) to a CSV file. If out_path is not given, writes
        to <root>/data/vrcft_eyedata_full_<participant_name>.csv.
        Returns the absolute path of the written file.
        """
        if not self.data_loaded or self.raw_data is None:
            raise RuntimeError(f"Error: eyedata was not loaded for {self.participant_name} ({self.file_path}); nothing to export.")
        if out_path is None:
            out_path = os.path.join(self.project_root(), "data",
                                   f"vrcft_eyedata_full_{self.participant_name.lower()}.csv")
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        self.raw_data.to_csv(out_path, index=False)
        print(f"Exported {len(self.raw_data)} rows x {len(self.raw_data.columns)} columns -> {os.path.abspath(out_path)}")
        return os.path.abspath(out_path)
    #endregion


# testing below to see if it works
if __name__ == "__main__":
    # Single participant from the 2026 retreat dataset (Case 1, Hamish's PC)
    participant_rel_path = "Recordings/2026_Retreat_Dataset/Case 1/Hamishs PC/VRCFT_EyeData.csv"
    test = VrcftEyeDataLoader(participant_rel_path, participant_name="Hamish")
    if test.load_data():
        print(f"Loaded {len(test.timestamps)} samples for {test.participant_name} "
              f"({test.session_duration_seconds:.1f}s session); "
              f"columns: {list(test.raw_data.columns)}")
        test.export_full_csv()
    else:
        print("Failed to load the 2026 retreat participant eyedata.")