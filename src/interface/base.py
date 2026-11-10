"""Base class for components with a validated config."""
import typing as tp

from abc import ABC
from omegaconf import DictConfig


T_ConfigValue: tp.TypeAlias = int | float | str | bool | None
T_LiteralSpec: tp.TypeAlias = tp.Callable[[T_ConfigValue], bool]
T_ConfigSpec:  tp.TypeAlias = dict[str, "T_ConfigSpec"] | T_LiteralSpec


class Base(ABC):
    CONFIG_SPEC: T_ConfigSpec

    @classmethod
    def check_config(
        cls,
        config: DictConfig | T_ConfigValue,
        spec_override: T_ConfigSpec | None = None,
    ) -> None:
        spec: T_ConfigSpec = cls.CONFIG_SPEC if spec_override is None else spec_override

        if isinstance(spec, dict):
            if not isinstance(config, DictConfig):
                raise TypeError(f"Config shape mismatch: expected a nested dict at this level, got literal value {config!r}")

            spec_keys: set[str] = set(spec.keys())
            cfg_keys:  set[str] = {str(k) for k in config.keys()}

            missing = spec_keys - cfg_keys
            if missing:
                raise KeyError(f"Config missing keys: {sorted(missing)}")
            extra = cfg_keys - spec_keys
            if extra:
                raise KeyError(f"Excessive Keys: {sorted(extra)} (allowed: {sorted(spec_keys)})")

            for key, sub_spec in spec.items():
                cls.check_config(config[key], sub_spec)

        else:
            if isinstance(config, DictConfig):
                raise TypeError(f"Config shape mismatch: expected a literal value, got DictConfig {config!r}")
            
            if not spec(config):
                raise ValueError(f"Config value {config!r} failed spec predicate {spec!r}")
