def validate_inputs(raw):
    errors = []

    for key, value in raw.items():
        if value is None:
            errors.append(f"{key} is missing.")
        if not isinstance(value, (int, float)):
            errors.append(f"{key} must be a number.")
        if value < 0:
            errors.append(f"{key} cannot be negative.")
    
    return errors
