"""
Constants for applefoundationmodels.

Provides default values and presets for generation parameters.
"""

# Generation defaults
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1024

# Temperature bounds
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


class TemperaturePreset:
    """
    Common temperature presets for different use cases.

    Temperature controls randomness in generation:
    - Lower values (0.1-0.3): More deterministic, good for facts and precision
    - Medium values (0.5-0.7): Balanced creativity and consistency
    - Higher values (1.0-1.5): More creative and varied outputs
    """

    DETERMINISTIC = 0.1  # Very low randomness, highly consistent
    FACTUAL = 0.3  # Low randomness, good for factual responses
    BALANCED = 0.7  # Balanced creativity and consistency
    CREATIVE = 1.0  # More creative responses
    VERY_CREATIVE = 1.5  # High creativity and variety
