"""
Custom logging formatter that includes currency pair information in log messages.
"""

import logging


class CurrencyFormatter(logging.Formatter):
    """
    Custom formatter that includes currency pair information in log messages.
    """
    
    def format(self, record):
        # Get the base formatted message
        msg = super().format(record)
        
        # Add currency pair information if available
        currency_info = ""
        if hasattr(record, 'pair') and record.pair:
            currency_info = f" [MOEDA: {record.pair}]"
        elif hasattr(record, 'currency_pair') and record.currency_pair:
            currency_info = f" [MOEDA: {record.currency_pair}]"
        else:
            # Try to get currency pair from context
            try:
                from .log_context import get_context
                context = get_context()
                if context.get('pair'):
                    currency_info = f" [MOEDA: {context['pair']}]"
            except Exception:
                pass
        
        return msg + currency_info


class CurrencyConsoleFormatter(CurrencyFormatter):
    """
    Console formatter with currency pair information.
    """
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
            datefmt='%H:%M:%S'
        )
