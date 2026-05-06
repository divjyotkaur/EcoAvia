"""Passenger Carbon Efficiency Score (PCES) Calculator."""
import config


class PCESCalculator:
    """PCES = (BASELINE_CO2 / adjusted_co2_per_pax) × 100."""

    def calculate(self, co2_per_pax_kg, load_factor=None, saf_blend_pct=0.0):
        """
        Args:
            co2_per_pax_kg: estimated CO2 kg per passenger
            load_factor: aircraft load factor (0-1)
            saf_blend_pct: SAF blend percentage (0-1)

        Returns:
            dict with PCES score and rating
        """
        if load_factor is None:
            load_factor = config.DEFAULT_LOAD_FACTOR

        # SAF reduces CO2
        adjusted_co2 = co2_per_pax_kg * (1 - saf_blend_pct * config.SAF_EMISSION_REDUCTION_FACTOR)

        # Load factor adjustment: higher load = better efficiency
        efficiency_adjusted = adjusted_co2 / load_factor

        # Score: 100 = at baseline
        pces = (config.BASELINE_CO2_PER_PAX_KG / efficiency_adjusted) * 100 if efficiency_adjusted > 0 else 0

        if pces > 110:
            rating = 'Excellent'
        elif pces > 95:
            rating = 'Good'
        elif pces > 80:
            rating = 'Fair'
        else:
            rating = 'Poor'

        return {
            'pces_score': round(pces, 1),
            'adjusted_co2_per_pax': round(efficiency_adjusted, 2),
            'rating': rating
        }
