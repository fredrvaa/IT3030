def yes_or_no(yn: str) -> bool:
    s = yn.lower().strip()
    return True if s in ['y', 'yes', 'ye'] else False
