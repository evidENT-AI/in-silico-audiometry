"""
NHANES-specific constants for data downloading and processing.

Provides URLs, component types, and standard configurations for
accessing NHANES data from CDC website.
"""

from typing import Dict, List

# ============================================================================
# NHANES Website URLs
# ============================================================================

NHANES_BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes"

# Component-specific data page URLs
NHANES_COMPONENT_URLS: Dict[str, str] = {
    "Demographics": f"{NHANES_BASE_URL}/search/datapage.aspx?Component=Demographics",
    "Dietary": f"{NHANES_BASE_URL}/search/datapage.aspx?Component=Dietary",
    "Examination": f"{NHANES_BASE_URL}/search/datapage.aspx?Component=Examination",
    "Laboratory": f"{NHANES_BASE_URL}/search/datapage.aspx?Component=Laboratory",
    "Questionnaire": f"{NHANES_BASE_URL}/search/datapage.aspx?Component=Questionnaire",
}

# ============================================================================
# NHANES Components
# ============================================================================

NHANES_COMPONENTS: List[str] = [
    "Demographics",
    "Dietary",
    "Examination",
    "Laboratory",
    "Questionnaire",
]

# ============================================================================
# Audiometric Dataset Names (in Examination component)
# ============================================================================

AUDIOMETRY_DATASETS: List[str] = [
    "Audiometry",
    "Audiometry - Acoustic Reflex",
    "Audiometry - Tympanometry",
    "Audiometry - Wideband Reflectance",
]

# Datasets needed for prior construction
PRIOR_DATASETS: Dict[str, List[str]] = {
    "Examination": [
        "Audiometry",
        "Audiometry - Tympanometry",
    ],
    "Demographics": [],  # Will download all demographics
    "Questionnaire": [
        "Diabetes",
        "Medical Conditions",
        "Blood Pressure & Cholesterol",
        "Audiometry",  # Audiometry questionnaire (different from examination)
    ],
}

# ============================================================================
# NHANES Year Cycles
# ============================================================================

# All NHANES cycles with audiometric data
NHANES_AUDIO_CYCLES: List[str] = [
    "1999-2000",
    "2001-2002",
    "2003-2004",
    "2005-2006",
    "2009-2010",
    "2011-2012",
    "2015-2016",
    "2017-2018",
    "2017-2020",
]

# Recommended cycles for prior construction (most recent with good data)
RECOMMENDED_CYCLES: List[str] = [
    "2015-2016",
    "2017-2018",
]

# ============================================================================
# Audiometric Frequency Columns
# ============================================================================

# NHANES audiometry variable codes for pure-tone thresholds
# Format: AUXU{freq}R/L where freq is frequency in Hz (abbreviated)
PTA_COLUMNS: Dict[str, Dict[str, str]] = {
    500: {"right": "AUXU500R", "left": "AUXU500L"},
    1000: {"right": "AUXU1K1R", "left": "AUXU1K1L"},
    2000: {"right": "AUXU2KR", "left": "AUXU2KL"},
    3000: {"right": "AUXU3KR", "left": "AUXU3KL"},
    4000: {"right": "AUXU4KR", "left": "AUXU4KL"},
    6000: {"right": "AUXU6KR", "left": "AUXU6KL"},
    8000: {"right": "AUXU8KR", "left": "AUXU8KL"},
}

# Standard audiometric frequencies (Hz)
STANDARD_FREQUENCIES: List[int] = [250, 500, 1000, 2000, 4000, 8000]

# Extended frequencies available in NHANES
NHANES_FREQUENCIES: List[int] = [500, 1000, 2000, 3000, 4000, 6000, 8000]

# ============================================================================
# Demographics Columns
# ============================================================================

DEMO_COLUMNS: Dict[str, str] = {
    "SEQN": "Respondent sequence number",
    "RIDAGEYR": "Age in years at screening",
    "RIAGENDR": "Gender (1=Male, 2=Female)",
    "RIDRETH3": "Race/ethnicity (detailed)",
}

# ============================================================================
# Questionnaire Columns for Prior Conditioning
# ============================================================================

# Diabetes questionnaire
DIABETES_COLUMNS: Dict[str, str] = {
    "DIQ010": "Doctor told you have diabetes",
    "DIQ050": "Taking insulin now",
    "DIQ070": "Taking diabetic pills",
}

# Medical conditions
MEDICAL_COLUMNS: Dict[str, str] = {
    "MCQ160B": "Ever told had congestive heart failure",
    "MCQ160C": "Ever told had coronary heart disease",
    "MCQ160D": "Ever told had angina/angina pectoris",
    "MCQ160E": "Ever told had heart attack",
    "MCQ160F": "Ever told had stroke",
}

# Blood pressure/cholesterol
BP_COLUMNS: Dict[str, str] = {
    "BPQ020": "Ever told you had high blood pressure",
    "BPQ080": "Doctor told you - Loss of BP control",
}

# Audiometry questionnaire
AUQ_COLUMNS: Dict[str, str] = {
    "AUQ054": "General condition of hearing",
    "AUQ060": "Hear a whisper from across a quiet room",
    "AUQ070": "Hear normal voice across a quiet room",
    "AUQ080": "How often do you have ringing ears",
    "AUQ090": "How long bothered by ringing, roaring",
    "AUQ191": "Ears ringing, roaring, buzzing past year",
    "AUQ250": "Ever worked 3+ months loud noise",
    "AUQ280": "Loud noise from guns or firearms",
}

# ============================================================================
# Tympanometry Columns
# ============================================================================

TYMP_COLUMNS: Dict[str, str] = {
    "AUXTPVR": "Peak pressure right ear",
    "AUXTPVL": "Peak pressure left ear",
    "AUXTCOMR": "Compliance right ear",
    "AUXTCOML": "Compliance left ear",
    "AUXTMETR": "Middle ear type right (A/As/Ad/B/C)",
    "AUXTMETL": "Middle ear type left (A/As/Ad/B/C)",
}

# ============================================================================
# Key Columns
# ============================================================================

# Primary key column for participant identification
SEQN_COLUMN = "SEQN"
SEQN_COLUMN_READABLE = "SEQN - Respondent sequence number"

# ============================================================================
# File Patterns and Extensions
# ============================================================================

# Valid data file extensions
VALID_EXTENSIONS = [".xpt", ".zip"]

# Files to exclude from processing
EXCLUDE_PATTERNS = ["mapping", "readme", "doc"]

# ============================================================================
# Output Directory Structure
# ============================================================================

# Subdirectories for organized data storage
DEFAULT_SUBDIRS = {
    "xpt": "xpt",           # Raw XPT files
    "csv": "csv",           # Converted CSV files
    "readable": "readable",  # Human-readable CSV with labels
    "merged": "merged",     # Merged datasets
    "priors": "priors",     # Computed prior distributions
}

# ============================================================================
# HTTP/FTP Configuration
# ============================================================================

# Download chunk size for streaming
DOWNLOAD_CHUNK_SIZE = 8192  # 8 KB

# HTTP request timeout (seconds)
REQUEST_TIMEOUT = 30

# Maximum retries for failed downloads
MAX_RETRIES = 3

# ============================================================================
# HTML Parsing Constants
# ============================================================================

# Expected table headers on NHANES data pages
EXPECTED_TABLE_HEADERS = ["Years", "Data File Name", "Data File", "Doc File"]

# CSS selectors for documentation parsing
DOC_VAR_TITLE_CLASS = "vartitle"

# ============================================================================
# Data Quality Constants
# ============================================================================

# Encoding options to try when reading/writing CSV
CSV_ENCODINGS = ["utf-8", "utf-8-sig", "latin1"]

# Minimum rows expected in a valid dataset
MIN_VALID_ROWS = 1

# ============================================================================
# Prior Construction Constants
# ============================================================================

# Age groups for stratified priors
AGE_GROUPS: List[tuple] = [
    (18, 39),
    (40, 59),
    (60, 79),
    (80, 120),
]

# Threshold bounds for valid audiometric data (dB HL)
THRESHOLD_BOUNDS = {
    "min": -10,
    "max": 120,
}

# KDE bandwidth (Scott's rule will be used by default, this is fallback)
DEFAULT_KDE_BANDWIDTH = 3.0  # dB
