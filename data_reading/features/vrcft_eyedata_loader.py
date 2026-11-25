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
    def load_data(self,
                  sep: str = ",",
                  filter_interaction_window: bool = False,
                  dyad_name: str = None,
                  participant_label: str = None,
                  time_reference: str = "utc",
                  padding_ms: float = 500.0) -> bool:
        """
        Loads the per-participant eyedata CSV.

        Parameters
        ----------
        sep : str
            CSV field separator.
        filter_interaction_window : bool
            If ``True``, samples outside the participant's interaction window
            (``2026_retreat_dyads_interaction_times.csv``) are dropped during
            load, so ``raw_data``, ``timestamps`` and ``session_duration_seconds``
            already reflect the window. Requires ``dyad_name``.
        dyad_name : str
            ``name`` value in the interaction CSV (e.g. ``"DYAD_2026_10_22_Eng_Case_1"``).
            Only used when ``filter_interaction_window`` is ``True``.
        participant_label : str
            ``participant`` value in the interaction CSV (``"p1"``/``"p2"``).
            Defaults to the lowercased ``participant_name`` (e.g. ``"P1"`` -> ``"p1"``).
        time_reference : str
            ``"utc"`` or ``"ntp"``; which boundary columns define the window.
        padding_ms : float
            Margin added on each side of the window boundaries.
        """
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
        # drop the raw hex timestamp column; we only use TimestampUtc/NtpOffsetMs
        if "NtpTimestampHex" in df.columns:
            df = df.drop(columns=["NtpTimestampHex"])
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
        if filter_interaction_window:
            if dyad_name is None:
                raise ValueError(
                    "filter_interaction_window=True requires dyad_name "
                    "(e.g. 'DYAD_2026_10_22_Eng_Case_1')."
                )
            label = participant_label or self.participant_name.lower()
            self.filter_by_interaction_window(
                dyad_name, label, time_reference=time_reference, padding_ms=padding_ms
            )
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

    #region interaction-window filtering
    def filter_by_interaction_window(
        self,
        dyad_name: str,
        participant_label: str = "p1",
        time_reference: str = "utc",  # "utc" or "ntp"
        padding_ms: float = 500.0,
    ) -> "VrcftEyeDataLoader":
        """
        Drops samples that fall outside the interaction window for a given dyad
        and participant, as defined in ``2026_retreat_dyads_interaction_times.csv``.

        The CSV is loaded once from *<project_root>/data/* and cached on the class
        so repeated calls (e.g. for p1 / p2 of the same dyad) stay fast.

        Parameters
        ----------
        dyad_name : str
            Exact ``name`` value in the CSV, e.g. ``"DYAD_2026_10_22_Eng_Case_1"``.
        participant_label : str
            ``participant`` column value in the CSV (e.g. ``"p1"``, ``"p2"``).
        time_reference : str
            Which boundary to use for filtering: ``"utc"`` or ``"ntp"``.
            Defaults to ``"utc"``: the UTC boundaries in the CSV have been
            verified correct against the dataset, and the loaded ``ts`` column
            is UTC-derived. Use ``"ntp"`` only if the device's NTP clock is
            known to be in sync with the interaction-time logger.
        padding_ms : float
            Extra margin (on each side) in milliseconds. The CSV start/end will
            not align exactly to sample boundaries; this avoids discarding valid
            edge samples that are within a few ms of the window boundary.

        Returns
        -------
        self  (for method chaining).
        """
        if not self.data_loaded or self.raw_data is None:
            raise RuntimeError(
                f"Cannot filter: eyedata was not loaded for {self.participant_name} "
                f"({self.file_path}). Call load_data() first."
            )

        # --- load and cache the interaction-time CSV ---------------------------
        if not hasattr(VrcftEyeDataLoader, "_interaction_cache"):
            root = VrcftEyeDataLoader.project_root()
            csv_path = os.path.join(root, "data", "2026_retreat_dyads_interaction_times.csv")
            try:
                VrcftEyeDataLoader._interaction_cache = pd.read_csv(csv_path)
                # header has a stray space (' participant'); normalize column names
                VrcftEyeDataLoader._interaction_cache.columns = [
                    str(c).strip() for c in VrcftEyeDataLoader._interaction_cache.columns
                ]
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Interaction-time CSV not found at {csv_path}. "
                    "Is the 2026 retreat dataset present?"
                )

        cache = VrcftEyeDataLoader._interaction_cache
        mask = (cache["name"] == dyad_name) & (cache["participant"] == participant_label)
        if not mask.any():
            raise ValueError(
                f"No interaction window found for dyad={dyad_name}, "
                f"participant={participant_label} in the CSV."
            )

        row = cache.loc[mask].iloc[0]

        # --- pick start / end -------------------------------------------------
        if time_reference == "ntp":
            ts_key = "ts_ntp"
            start_raw = pd.to_datetime(row["start_interaction_ntp"], utc=True).tz_localize(None)
            end_raw   = pd.to_datetime(row["end_interaction_ntp"],   utc=True).tz_localize(None)
        else:
            ts_key = "ts"
            start_raw = pd.to_datetime(row["start_interaction_utc"], utc=True).tz_localize(None)
            end_raw   = pd.to_datetime(row["end_interaction_utc"],   utc=True).tz_localize(None)

        start_trim = start_raw - pd.Timedelta(milliseconds=padding_ms)
        end_trim   = end_raw   + pd.Timedelta(milliseconds=padding_ms)

        # --- filter the raw data ------------------------------------------------
        ts_col = self.raw_data[ts_key]
        before = ts_col >= start_trim
        after  = ts_col <= end_trim
        kept_mask = before & after
        dropped = (~kept_mask).sum()

        if dropped > 0:
            self.raw_data = self.raw_data.loc[kept_mask].reset_index(drop=True)
            self.timestamps = self.raw_data["ts"].tolist()
            # Recalculate session duration from the trimmed data
            self.session_duration_seconds = (
                self.timestamps[-1] - self.timestamps[0]
            ).total_seconds()
            print(
                f"[filter] Dyad={dyad_name}, {participant_label}: "
                f"dropped {dropped}/{len(self.raw_data) + dropped} samples "
                f"(window: {start_raw.time()} – {end_raw.time()} "
                f"on '{time_reference}', pad={padding_ms:.0f}ms, "
                f"remaining={len(self.raw_data)} frames, "
                f"duration={self.session_duration_seconds:.1f}s)"
            )
        else:
            print(
                f"[filter] Dyad={dyad_name}, {participant_label}: "
                f"all {len(self.raw_data)} samples inside window "
                f"(no change)."
            )

        return self  # enable chaining
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
    # Single participant from the 2026 retreat dataset (Case 1, Hamish's PC).
    # Hamish is 'p1' of DYAD_2026_10_22_Eng_Case_1 in the interaction CSV.
    participant_rel_path = "Recordings/2026_Retreat_Dataset/Case 1/Hamishs PC/VRCFT_EyeData.csv"
    dyad_name = "DYAD_2026_10_22_Eng_Case_1"

    # 1) raw load (flag off, default behaviour)
    test = VrcftEyeDataLoader(participant_rel_path, participant_name="Hamish")
    if not test.load_data():
        raise SystemExit("Failed to load the 2026 retreat participant eyedata.")
    raw_samples = len(test.raw_data)
    raw_duration = test.session_duration_seconds
    print(f"Raw:       {raw_samples} samples, {raw_duration:.1f}s; "
          f"columns: {list(test.raw_data.columns)}")
    #test.export_full_csv()

    # 2) filtered load (restricted to the interaction window, UTC boundaries)
    filtered = VrcftEyeDataLoader(participant_rel_path, participant_name="Hamish")
    filtered.load_data(
        filter_interaction_window=True,
        dyad_name=dyad_name,
        participant_label="p1",   # participant_name 'Hamish' is not a CSV label
    )
    dropped = raw_samples - len(filtered.raw_data)
    print(f"Filtered:  {len(filtered.raw_data)} samples, "
          f"{filtered.session_duration_seconds:.1f}s ({dropped} dropped) "
          f"for {dyad_name} / p1")

    # 3) blink detection on the filtered data
    onsets, durations = filtered.detect_blinks()
    print(f"Blinks:    {len(onsets)} valid blinks, "
          f"mean duration {filtered.mean_blink_duration_ms():.0f} ms, "
          f"rate {len(onsets) / (filtered.session_duration_seconds / 60.0):.2f}/min")