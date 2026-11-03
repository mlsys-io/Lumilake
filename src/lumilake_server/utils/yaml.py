"""YAML dumping with libyaml when PyYAML provides it."""

from typing import Any

import yaml

_Dumper = yaml.CSafeDumper if yaml.__with_libyaml__ else yaml.SafeDumper


def dump_yaml(value: Any, **kwargs: Any) -> str:
    return yaml.dump(value, Dumper=_Dumper, **kwargs)
