import logging


def record_args(logger: logging.Logger, args) -> None:
    """Record all parameters from args to the logger."""
    logger.info("=" * 60)
    logger.info("Parameters:")
    logger.info("=" * 60)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
