import logging
import sys
import yaml

from box import Box
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_project_root() -> Path:
    """
    Finds the project root by searching upwards for a marker file.
    Raises FileNotFoundError if no root marker is found.
    """

    current_path = Path(__file__).resolve().parent
    root_markers = ["pyproject.toml", "config.yaml", ".git"]

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in root_markers):
            return parent

    raise FileNotFoundError(
        f"Could not find project root. Looked for {root_markers} "
        f"in {current_path} or any of its parent directories."
    )


def load_config(config_path: Path):
    """
    Loads the YAML configuration file.
    """

    with open(config_path, "r") as f:
        return Box(yaml.safe_load(f))


def configure_logger(level: str = "INFO", fmt: Optional[str] = None):
    """
    Configures or re-configures the global logging system.
    """

    if fmt is None:
        fmt = "%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    return logging.getLogger(__name__)


def validate_date_interval(start_date: str, end_date: str, date_format: str):
    """
    Validates date strings and ensures logical order.
    Raises ValueError if validation fails.
    """

    try:
        start = datetime.strptime(start_date, date_format)
        end = datetime.strptime(end_date, date_format)
    except ValueError:
        raise ValueError(
            f"Incorrect dates' format:\n"
            f"Current values: '{start_date}', '{end_date}'\n"
            f"Expected format: '{date_format}'"
        )

    if start > end:
        raise ValueError(f"Start date ({start_date}) cannot be after end date ({end_date})")

    if end > datetime.now():
        raise ValueError(f"End date ({end_date}) cannot be in the future.")
