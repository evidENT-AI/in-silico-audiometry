"""
NHANES Data Downloader Module

Downloads and converts NHANES datasets from CDC website.
Based on work by Isabella Bellomo, adapted for audiometry_ai.

Features:
- Scrapes NHANES data pages by component
- Downloads XPT files (HTTP/FTP support)
- Converts XPT to CSV using pyreadstat
- Extracts variable labels from documentation
- Creates human-readable CSVs with labeled columns
- Supports selective dataset downloading
"""

import os
import re
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from ftplib import FTP

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..utils.logging import setup_logger
from ..utils.io import ensure_directory_exists
from .constants import (
    NHANES_COMPONENT_URLS,
    DOWNLOAD_CHUNK_SIZE,
    REQUEST_TIMEOUT,
    DOC_VAR_TITLE_CLASS,
    DEFAULT_SUBDIRS,
    CSV_ENCODINGS,
)

logger = setup_logger(__name__)


class NHANESDownloader:
    """
    Download and convert NHANES datasets from CDC website.

    This class handles the complete workflow of downloading NHANES data:
    1. Scraping data pages to find available datasets
    2. Downloading XPT files (via HTTP or FTP)
    3. Converting XPT to CSV format
    4. Parsing documentation to extract variable labels
    5. Creating human-readable CSV files with labeled columns

    Parameters
    ----------
    output_dir : str or Path, optional
        Base directory for downloaded data. Defaults to 'data/nhanes'
        relative to project root.
    component : str, optional
        NHANES component to download from. Options: Demographics, Dietary,
        Examination, Laboratory, Questionnaire. Default: "Examination"
    create_readable : bool, default=True
        Whether to create human-readable CSV files with column labels
    overwrite : bool, default=False
        Whether to re-download existing files
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        component: str = "Examination",
        create_readable: bool = True,
        overwrite: bool = False
    ):
        # Default to data/nhanes relative to project root
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "nhanes"

        self.output_dir = Path(output_dir)
        self.component = component
        self.create_readable = create_readable
        self.overwrite = overwrite

        # Create subdirectories
        self.xpt_dir = self.output_dir / DEFAULT_SUBDIRS["xpt"]
        self.csv_dir = self.output_dir / DEFAULT_SUBDIRS["csv"]
        self.readable_dir = self.output_dir / DEFAULT_SUBDIRS["readable"]
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.xpt_dir, self.csv_dir, self.readable_dir, self.metadata_dir]:
            ensure_directory_exists(dir_path)

        logger.info(f"NHANESDownloader initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Component: {self.component}")

    def download_datasets(
        self,
        datasets: Optional[List[str]] = None,
        years: Optional[List[str]] = None
    ) -> Dict[str, List[Path]]:
        """
        Download specified NHANES datasets.

        Parameters
        ----------
        datasets : list of str, optional
            List of dataset names to download. If None, downloads all available.
        years : list of str, optional
            List of year ranges to download (e.g., ["2015-2016", "2017-2018"]).
            If None, downloads all years.

        Returns
        -------
        dict
            Dictionary mapping dataset names to lists of downloaded CSV files
        """
        base_url = NHANES_COMPONENT_URLS.get(self.component)
        if not base_url:
            raise ValueError(f"Unknown component: {self.component}")

        logger.info(f"Fetching datasets from {base_url}")

        # Parse the NHANES data page
        dataset_info = self._parse_data_page(base_url)

        # Filter by requested datasets and years
        if datasets:
            dataset_info = {k: v for k, v in dataset_info.items() if k in datasets}

        if years:
            years_set = set(years)
            for dataset_name in list(dataset_info.keys()):
                dataset_info[dataset_name] = [
                    item for item in dataset_info[dataset_name]
                    if item['years'] in years_set
                ]
                if not dataset_info[dataset_name]:
                    del dataset_info[dataset_name]

        logger.info(f"Found {len(dataset_info)} dataset(s) to download")

        # Download each dataset
        results = {}
        for dataset_name, items in dataset_info.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*60}")

            csv_files = self._download_dataset(dataset_name, items, base_url)
            results[dataset_name] = csv_files

        logger.info(f"\n✅ Download complete! Processed {len(results)} dataset(s)")
        return results

    def _parse_data_page(self, url: str) -> Dict[str, List[Dict]]:
        """Parse NHANES data page to extract dataset information."""
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")

        if not table:
            raise ValueError("No data table found on page")

        rows = table.find_all("tr")
        if not rows:
            raise ValueError("No rows found in data table")

        # Parse header
        header = [th.get_text(strip=True) for th in rows[0].find_all("th")]
        logger.info(f"Table columns: {header}")

        # Get column indices
        try:
            year_idx = header.index("Years")
            name_idx = header.index("Data File Name")
            file_idx = header.index("Data File")
            doc_idx = header.index("Doc File")
        except ValueError as e:
            logger.error(f"Missing expected column in table: {e}")
            raise

        # Parse data rows
        dataset_info = {}
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) <= max(year_idx, name_idx, file_idx, doc_idx):
                continue

            dataset_name = cols[name_idx].get_text(strip=True)
            dataset_years = cols[year_idx].get_text(strip=True)

            if not dataset_name:
                continue

            # Extract file links
            file_links = cols[file_idx].find_all("a")
            doc_links = cols[doc_idx].find_all("a")

            data_files = [link.get("href") for link in file_links if link.get("href")]
            doc_url = doc_links[0].get("href") if doc_links else None

            if dataset_name not in dataset_info:
                dataset_info[dataset_name] = []

            dataset_info[dataset_name].append({
                'years': dataset_years,
                'data_files': data_files,
                'doc_url': doc_url
            })

        return dataset_info

    def _download_dataset(
        self,
        dataset_name: str,
        items: List[Dict],
        base_url: str
    ) -> List[Path]:
        """Download all files for a specific dataset."""
        # Create dataset directories
        dataset_xpt = self.xpt_dir / self._sanitize_filename(dataset_name)
        dataset_csv = self.csv_dir / self._sanitize_filename(dataset_name)
        dataset_readable = self.readable_dir / self._sanitize_filename(dataset_name)

        for dir_path in [dataset_xpt, dataset_csv, dataset_readable]:
            ensure_directory_exists(dir_path)

        csv_files = []

        for item in items:
            years = item['years']
            doc_url = item['doc_url']

            # Parse documentation for variable labels
            variable_mapping = {}
            comprehensive_metadata = {}
            if doc_url and self.create_readable:
                full_doc_url = urljoin(base_url, doc_url)
                variable_mapping = self._parse_doc_page(full_doc_url)

                if variable_mapping:
                    # Save mapping file
                    map_df = pd.DataFrame(
                        list(variable_mapping.items()),
                        columns=["Code", "Label"]
                    )
                    map_path = dataset_readable / "variable_mapping.csv"
                    map_df.to_csv(map_path, index=False, encoding="utf-8")
                    logger.info(f"Saved variable mapping: {map_path}")

                # Parse comprehensive metadata for JSON
                comprehensive_metadata = self._parse_comprehensive_metadata(full_doc_url)

            # Download data files
            for file_url in item['data_files']:
                full_url = urljoin(base_url, file_url)
                csv_file = self._download_and_convert_file(
                    url=full_url,
                    dataset_name=dataset_name,
                    years=years,
                    xpt_dir=dataset_xpt,
                    csv_dir=dataset_csv,
                    readable_dir=dataset_readable,
                    variable_mapping=variable_mapping
                )
                if csv_file:
                    csv_files.append(csv_file)

                    # Create JSON metadata file
                    if comprehensive_metadata and doc_url:
                        try:
                            dataset_code = csv_file.stem.split('_')[-1]
                            self._create_dataset_metadata_json(
                                dataset_name=dataset_name,
                                dataset_code=dataset_code,
                                year_cycle=years,
                                csv_path=csv_file,
                                doc_url=urljoin(base_url, doc_url),
                                variable_metadata=comprehensive_metadata
                            )
                        except Exception as e:
                            logger.warning(f"Failed to create metadata JSON: {e}")

        return csv_files

    def _download_and_convert_file(
        self,
        url: str,
        dataset_name: str,
        years: str,
        xpt_dir: Path,
        csv_dir: Path,
        readable_dir: Path,
        variable_mapping: Dict[str, str]
    ) -> Optional[Path]:
        """Download and convert a single NHANES data file."""
        filename = os.path.basename(urlparse(url).path)
        base_name = os.path.splitext(filename)[0]

        # Create sanitized filenames
        years_clean = self._sanitize_filename(years)
        dataset_clean = self._sanitize_filename(dataset_name)
        new_base_name = f"{years_clean}_{dataset_clean}_{base_name}"

        xpt_path = xpt_dir / filename
        csv_path = csv_dir / f"{new_base_name}.csv"
        readable_path = readable_dir / f"{new_base_name}_readable.csv"

        # Download XPT file if needed
        if not xpt_path.exists() or self.overwrite:
            try:
                self._download_file(url, xpt_path)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                return None
        else:
            logger.info(f"File already exists: {xpt_path.name}")

        # Handle ZIP files
        if xpt_path.suffix.lower() == ".zip":
            return self._process_zip_file(
                xpt_path, years_clean, dataset_clean,
                csv_dir, readable_dir, variable_mapping
            )

        # Handle XPT files
        elif xpt_path.suffix.lower() == ".xpt":
            # Convert to CSV
            if not csv_path.exists() or self.overwrite:
                df = self._convert_xpt_to_csv(xpt_path, csv_path)
            else:
                logger.info(f"CSV already exists: {csv_path.name}")
                try:
                    df = pd.read_csv(csv_path, low_memory=False)
                except Exception as e:
                    logger.warning(f"Could not read existing CSV: {e}")
                    df = self._convert_xpt_to_csv(xpt_path, csv_path)

            # Create readable version
            if df is not None and variable_mapping and self.create_readable:
                if not readable_path.exists() or self.overwrite:
                    self._create_readable_csv(df, readable_path, variable_mapping)

            return csv_path if csv_path.exists() else None

        return None

    def _process_zip_file(
        self,
        zip_path: Path,
        years_clean: str,
        dataset_clean: str,
        csv_dir: Path,
        readable_dir: Path,
        variable_mapping: Dict[str, str]
    ) -> Optional[Path]:
        """Extract and process XPT files from ZIP archive."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_path.parent)

                for extracted_file in zip_ref.namelist():
                    if extracted_file.lower().endswith(".xpt"):
                        extracted_path = zip_path.parent / extracted_file
                        extracted_base = os.path.splitext(extracted_file)[0]
                        extracted_clean = self._sanitize_filename(extracted_base)

                        csv_name = f"{years_clean}_{dataset_clean}_{extracted_clean}.csv"
                        csv_path = csv_dir / csv_name

                        # Convert to CSV
                        df = self._convert_xpt_to_csv(extracted_path, csv_path)

                        # Create readable version
                        if df is not None and variable_mapping and self.create_readable:
                            readable_name = f"{years_clean}_{dataset_clean}_{extracted_clean}_readable.csv"
                            readable_path = readable_dir / readable_name
                            self._create_readable_csv(df, readable_path, variable_mapping)

                        return csv_path
        except Exception as e:
            logger.error(f"Failed to process ZIP file {zip_path}: {e}")

        return None

    def _download_file(self, url: str, save_path: Path) -> None:
        """Download a file from URL (HTTP or FTP)."""
        parsed = urlparse(url)

        if parsed.scheme == "ftp":
            logger.info(f"Downloading via FTP: {save_path.name}")
            try:
                ftp = FTP(parsed.hostname)
                ftp.login()
                ftp.cwd(os.path.dirname(parsed.path))

                with open(save_path, "wb") as f:
                    ftp.retrbinary(f"RETR {os.path.basename(parsed.path)}", f.write)

                ftp.quit()
                logger.info(f"✓ Downloaded: {save_path.name}")
            except Exception as e:
                logger.error(f"FTP download failed: {e}")
                raise
        else:
            logger.info(f"Downloading via HTTP: {save_path.name}")
            try:
                response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

                logger.info(f"✓ Downloaded: {save_path.name}")
            except requests.RequestException as e:
                logger.error(f"HTTP download failed: {e}")
                raise

    def _convert_xpt_to_csv(self, xpt_path: Path, csv_path: Path) -> Optional[pd.DataFrame]:
        """Convert XPT file to CSV format."""
        try:
            import pyreadstat

            df, meta = pyreadstat.read_xport(str(xpt_path))

            # Try different encodings
            for encoding in CSV_ENCODINGS:
                try:
                    df.to_csv(csv_path, index=False, encoding=encoding)
                    logger.info(f"✓ Converted to CSV: {csv_path.name}")
                    return df
                except UnicodeEncodeError:
                    continue

            # If all encodings fail, use utf-8-sig
            df.to_csv(csv_path, index=False, encoding="utf-8-sig", errors="replace")
            logger.info(f"✓ Converted to CSV (with replacements): {csv_path.name}")
            return df

        except Exception as e:
            logger.error(f"Failed to convert {xpt_path.name}: {e}")
            return None

    def _create_readable_csv(
        self,
        df: pd.DataFrame,
        output_path: Path,
        variable_mapping: Dict[str, str]
    ) -> None:
        """Create human-readable CSV with labeled columns."""
        try:
            # Rename columns using mapping
            df_readable = df.rename(
                columns=lambda c: variable_mapping.get(c.upper(), c)
            )

            df_readable.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"✓ Created readable CSV: {output_path.name}")

        except Exception as e:
            logger.error(f"Failed to create readable CSV: {e}")

    def _parse_doc_page(self, doc_url: str) -> Dict[str, str]:
        """Parse NHANES documentation page to extract variable labels."""
        try:
            response = requests.get(doc_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            mapping = {}

            # Find variable title elements
            var_titles = soup.find_all("h3", class_=DOC_VAR_TITLE_CLASS)

            for h3 in var_titles:
                text = h3.get_text(strip=True)
                if " - " in text:
                    var_code, description = text.split(" - ", 1)
                    var_upper = var_code.strip().upper()
                    mapping[var_upper] = f"{var_upper} - {description.strip()}"

            logger.info(f"Parsed {len(mapping)} variable labels from documentation")
            return mapping

        except Exception as e:
            logger.warning(f"Failed to parse documentation page: {e}")
            return {}

    def _parse_comprehensive_metadata(self, doc_url: str) -> Dict[str, Dict[str, Any]]:
        """Parse comprehensive variable metadata from NHANES documentation page."""
        try:
            response = requests.get(doc_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            metadata = {}

            # Find all variable sections
            var_titles = soup.find_all("h3", class_=DOC_VAR_TITLE_CLASS)

            for h3 in var_titles:
                # Parse variable code and label
                text = h3.get_text(strip=True)
                if " - " not in text:
                    continue

                var_code, var_label = text.split(" - ", 1)
                var_code = var_code.strip().upper()
                var_label = var_label.strip()

                # Initialize variable metadata
                var_meta = {
                    "code": var_code,
                    "label": var_label,
                    "description": var_label,
                    "type": "unknown",
                    "units": "",
                    "values": {},
                    "notes": "",
                    "range": None
                }

                # Find the table following this variable (contains value codes)
                next_elem = h3.find_next_sibling()

                while next_elem:
                    # Check for value code table
                    if next_elem.name == "table":
                        rows = next_elem.find_all("tr")
                        if len(rows) > 1:
                            for row in rows[1:]:
                                cols = row.find_all("td")
                                if len(cols) >= 2:
                                    code = cols[0].get_text(strip=True)
                                    desc = cols[1].get_text(strip=True)
                                    var_meta["values"][code] = desc

                            if var_meta["values"]:
                                var_meta["type"] = "categorical"
                        break

                    elif next_elem.name == "div":
                        text_content = next_elem.get_text(strip=True)

                        if "unit" in text_content.lower() or "mm hg" in text_content.lower():
                            units_match = re.search(r'\((.*?)\)', text_content)
                            if units_match:
                                var_meta["units"] = units_match.group(1)

                        range_match = re.search(r'Range.*?(\d+\.?\d*)\s*to\s*(\d+\.?\d*)', text_content, re.IGNORECASE)
                        if range_match:
                            var_meta["range"] = {
                                "min": float(range_match.group(1)),
                                "max": float(range_match.group(2))
                            }
                            var_meta["type"] = "numeric"

                    elif next_elem.name == "h3":
                        break

                    next_elem = next_elem.find_next_sibling()

                # Infer type if still unknown
                if var_meta["type"] == "unknown":
                    if var_code == "SEQN":
                        var_meta["type"] = "identifier"
                    elif any(keyword in var_label.lower() for keyword in ["count", "number", "age", "level", "pressure", "weight", "height"]):
                        var_meta["type"] = "numeric"
                    elif any(keyword in var_label.lower() for keyword in ["text", "comment", "description"]):
                        var_meta["type"] = "text"

                metadata[var_code] = var_meta

            logger.info(f"Parsed comprehensive metadata for {len(metadata)} variables")
            return metadata

        except Exception as e:
            logger.warning(f"Failed to parse comprehensive metadata: {e}")
            return {}

    def _create_dataset_metadata_json(
        self,
        dataset_name: str,
        dataset_code: str,
        year_cycle: str,
        csv_path: Path,
        doc_url: str,
        variable_metadata: Dict[str, Dict[str, Any]]
    ) -> Path:
        """Create comprehensive JSON metadata file for a dataset."""
        try:
            df = pd.read_csv(csv_path, nrows=1)
            df_full = pd.read_csv(csv_path)
            n_rows = len(df_full)
            columns = list(df.columns)

            # Enhance metadata with actual data statistics
            for col in columns:
                col_upper = col.upper()
                if col_upper in variable_metadata:
                    if pd.api.types.is_numeric_dtype(df_full[col]):
                        variable_metadata[col_upper]["data_stats"] = {
                            "count": int(df_full[col].notna().sum()),
                            "missing": int(df_full[col].isna().sum()),
                            "missing_pct": float(df_full[col].isna().mean() * 100),
                            "min": float(df_full[col].min()) if df_full[col].notna().any() else None,
                            "max": float(df_full[col].max()) if df_full[col].notna().any() else None,
                            "mean": float(df_full[col].mean()) if df_full[col].notna().any() else None,
                            "median": float(df_full[col].median()) if df_full[col].notna().any() else None
                        }
                    else:
                        unique_values = df_full[col].nunique()
                        variable_metadata[col_upper]["data_stats"] = {
                            "count": int(df_full[col].notna().sum()),
                            "missing": int(df_full[col].isna().sum()),
                            "missing_pct": float(df_full[col].isna().mean() * 100),
                            "unique_values": int(unique_values)
                        }

        except Exception as e:
            logger.warning(f"Failed to compute data statistics: {e}")
            n_rows = 0
            columns = []

        # Create metadata structure
        metadata = {
            "dataset_info": {
                "name": dataset_name,
                "code": dataset_code,
                "year_cycle": year_cycle,
                "component": self.component,
                "documentation_url": doc_url,
                "download_date": datetime.now().isoformat(),
                "nhanes_version": year_cycle
            },
            "data_file": {
                "filename": csv_path.name,
                "path": str(csv_path.relative_to(self.output_dir)),
                "n_rows": n_rows,
                "n_columns": len(columns)
            },
            "variables": variable_metadata
        }

        # Save JSON metadata
        json_filename_base = f"{year_cycle}_{dataset_name.replace(' ', '_')}_{dataset_code}_metadata"
        json_filename_base = self._sanitize_filename(json_filename_base)
        json_filename = f"{json_filename_base}.json"
        json_path = self.metadata_dir / json_filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Created metadata JSON: {json_path.name}")
        return json_path

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename by removing invalid characters."""
        sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')
