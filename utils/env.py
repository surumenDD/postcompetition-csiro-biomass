from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    experiments_dir: Path = Path("experiments")
    postcompetition_dir: Path = Path("postcompetition")
    input_dir: Path = Path("data")
    output_dir: Path = Path("output/experiments")
