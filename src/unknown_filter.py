# import config

# def is_uncertain(confidence: float) -> bool:
#     """
#     Class-agnostic uncertainty check
#     """
#     return confidence < config.UNKNOWN_CONF_THRESHOLD
        
import config

def is_uncertain(confidence: float) -> bool:
    """
    Class-agnostic uncertainty check
    """
    return confidence < config.UNKNOWN_CONF_THRESHOLD