"""Prescriptive Recommendation Engine - Rule-based optimizer."""
import config


class PrescriptiveOptimizer:
    """Generate actionable recommendations from forecast data."""

    @staticmethod
    def generate_recommendations(forecast_dict, sustainability_dict):
        """
        Generate recommendations based on forecast outputs.

        Args:
            forecast_dict: dict with forecasts and metrics
            sustainability_dict: dict with ISI, PCES, CO2 data

        Returns:
            list of recommendation dicts
        """
        recommendations = []

        isi = sustainability_dict.get('isi', 0)
        pces_rating = sustainability_dict.get('pces_rating', 'Fair')
        co2_tonnes = sustainability_dict.get('co2_tonnes_annual', 0)

        # Rule 1: ISI-based fleet allocation
        if isi > config.ISI_CRITICAL_THRESHOLD:
            recommendations.append({
                'priority': 'Critical',
                'category': 'Fleet',
                'action': 'Wide-body reallocation required',
                'trigger': f'ISI={isi:.2f} exceeds critical threshold {config.ISI_CRITICAL_THRESHOLD}',
                'impact': 'Prevent gridlock, improve passenger experience'
            })
        elif isi > config.ISI_WARNING_THRESHOLD:
            recommendations.append({
                'priority': 'High',
                'category': 'Capacity',
                'action': 'Begin capacity expansion planning',
                'trigger': f'ISI={isi:.2f} approaching capacity limits',
                'impact': 'Proactive infrastructure scaling'
            })

        # Rule 2: Fuel hedging
        avg_fuel = forecast_dict.get('fuel', {}).get('prices', [2.5])
        if isinstance(avg_fuel, list) and len(avg_fuel) > 0:
            fuel_price = avg_fuel[-1] if avg_fuel else 2.5
        else:
            fuel_price = avg_fuel if isinstance(avg_fuel, (int, float)) else 2.5

        if fuel_price > config.FUEL_HEDGING_TRIGGER_PRICE:
            recommendations.append({
                'priority': 'High',
                'category': 'Fuel',
                'action': f'Activate fuel hedging ({int(config.FUEL_HEDGING_COVER_RATIO*100)}% forward cover)',
                'trigger': f'Jet fuel ${fuel_price:.2f}/gal > threshold ${config.FUEL_HEDGING_TRIGGER_PRICE}/gal',
                'impact': 'Lock in favorable fuel costs, mitigate price volatility'
            })

        # Rule 3: Sustainability actions
        if pces_rating in ['Poor', 'Fair']:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Sustainability',
                'action': 'Evaluate SAF blending strategy (5-10% SAF adoption)',
                'trigger': f'PCES rating: {pces_rating}',
                'impact': f'Reduce CO2 by ~{int(co2_tonnes * 0.05 * 0.8):,} tonnes/month (5% SAF blend)'
            })

        # Rule 4: Dynamic pricing
        recommendations.append({
            'priority': 'Standard',
            'category': 'Revenue',
            'action': 'Apply dynamic yield pricing (+2-4% premium)',
            'trigger': 'High passenger demand elasticity observed',
            'impact': 'Optimize revenue per available seat kilometer (RASM)'
        })

        return recommendations
