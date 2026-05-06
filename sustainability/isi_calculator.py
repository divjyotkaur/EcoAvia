"""Infrastructure Stress Index (ISI) Calculator."""
import config


class ISICalculator:
    """Calculate ISI = Projected Passengers / Terminal Design Capacity."""

    def calculate(self, projected_passengers_millions):
        """
        Args:
            projected_passengers_millions: annual projected passengers

        Returns:
            dict with ISI value, status, and capacity metrics
        """
        isi = projected_passengers_millions / config.SFO_TERMINAL_CAPACITY_MILLIONS

        if isi < config.ISI_NORMAL_THRESHOLD:
            status, color = 'Normal', 'green'
        elif isi <= config.ISI_WARNING_THRESHOLD:
            status, color = 'Warning', 'yellow'
        else:
            status, color = 'Critical', 'red'

        return {
            'isi': round(isi, 4),
            'status': status,
            'color': color,
            'capacity_used_pct': round(isi * 100, 1)
        }
