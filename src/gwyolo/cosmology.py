from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


C_KM_PER_SECOND = 299_792.458


@dataclass(frozen=True)
class FlatLambdaCDMGrid:
    """Deterministic flat-LambdaCDM distance grid without an Astropy dependency."""

    hubble_km_s_mpc: float = 67.66
    omega_matter: float = 0.3111
    maximum_redshift: float = 10.0
    points: int = 100_001

    def __post_init__(self) -> None:
        if self.hubble_km_s_mpc <= 0:
            raise ValueError("Hubble constant must be positive")
        if not 0 < self.omega_matter < 1:
            raise ValueError("Matter density must be between zero and one")
        if self.maximum_redshift <= 0 or self.points < 1001:
            raise ValueError("Cosmology grid is too small")

    @lru_cache(maxsize=8)
    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        redshift = np.linspace(0.0, self.maximum_redshift, self.points, dtype=np.float64)
        expansion = np.sqrt(
            self.omega_matter * (1.0 + redshift) ** 3 + (1.0 - self.omega_matter)
        )
        inverse_expansion = 1.0 / expansion
        spacing = redshift[1] - redshift[0]
        integral = np.empty_like(redshift)
        integral[0] = 0.0
        integral[1:] = np.cumsum(
            0.5 * (inverse_expansion[:-1] + inverse_expansion[1:]) * spacing
        )
        comoving = C_KM_PER_SECOND / self.hubble_km_s_mpc * integral
        luminosity = (1.0 + redshift) * comoving
        return redshift, comoving, luminosity

    def redshift_at_luminosity_distance(self, distance_mpc: float | np.ndarray) -> np.ndarray:
        redshift, _, luminosity = self.arrays()
        values = np.asarray(distance_mpc, dtype=np.float64)
        if np.any(values < 0) or np.any(values > luminosity[-1]):
            raise ValueError("Luminosity distance falls outside the cosmology grid")
        return np.interp(values, luminosity, redshift)

    def redshift_at_comoving_distance(self, distance_mpc: float | np.ndarray) -> np.ndarray:
        redshift, comoving, _ = self.arrays()
        values = np.asarray(distance_mpc, dtype=np.float64)
        if np.any(values < 0) or np.any(values > comoving[-1]):
            raise ValueError("Comoving distance falls outside the cosmology grid")
        return np.interp(values, comoving, redshift)

    def distances_at_redshift(
        self, redshift_value: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        redshift, comoving, luminosity = self.arrays()
        values = np.asarray(redshift_value, dtype=np.float64)
        if np.any(values < 0) or np.any(values > redshift[-1]):
            raise ValueError("Redshift falls outside the cosmology grid")
        return np.interp(values, redshift, comoving), np.interp(values, redshift, luminosity)

    def metadata(self) -> dict[str, float | int | str]:
        return {
            "model": "flat_lambda_cdm",
            "parameter_source": "Planck_2018_TT_TE_EE_lowE_lensing_BAO",
            "hubble_km_s_mpc": self.hubble_km_s_mpc,
            "omega_matter": self.omega_matter,
            "maximum_redshift": self.maximum_redshift,
            "grid_points": self.points,
        }
