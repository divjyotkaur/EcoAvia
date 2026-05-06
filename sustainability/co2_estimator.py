"""CO2 Emissions Estimator using ICAO standards."""
import config


class CO2Estimator:
    """Calculate CO2 emissions from passenger demand and fuel consumption."""

    def estimate(self, passengers_millions, fuel_price=None):
        """
        Args:
            passengers_millions: average monthly passengers in millions
            fuel_price: optional, for future hedging context

        Returns:
            dict with CO2 tonnes, fuel kg, and per-passenger metrics
        """
        total_pax = passengers_millions * 1_000_000

        # Fuel consumption: passengers × burn rate × distance / 100
        fuel_liters = (total_pax * config.FUEL_BURN_L_PER_100_KM_PER_PAX *
                      config.AVG_STAGE_LENGTH_KM / 100)
        fuel_kg = fuel_liters * config.JET_FUEL_DENSITY_KG_PER_LITER

        # CO2 = fuel × ICAO emission factor
        co2_kg = fuel_kg * config.ICAO_EMISSION_FACTOR_KG_PER_KG
        co2_tonnes = co2_kg / 1000

        return {
            'co2_tonnes': round(co2_tonnes, 0),
            'fuel_kg': round(fuel_kg, 0),
            'co2_per_pax_kg': round((co2_kg / total_pax) if total_pax > 0 else 0, 2)
        }
