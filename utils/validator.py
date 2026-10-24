"""
utils/validator.py
==================
Server-side validation for the PCOS screening questionnaire inputs.
Ensures strict type checking, valid numeric bounds, and permitted categorical values.
"""

# Allowed feature boundaries based on clinical bounds and training data distribution
FIELD_SPECS = {
    'Age (yrs)': {'type': float, 'min': 15.0, 'max': 60.0, 'required': True, 'label': 'Age'},
    'BMI': {'type': float, 'min': 10.0, 'max': 60.0, 'required': True, 'label': 'Body Mass Index (BMI)'},
    'Pulse rate(bpm)': {'type': float, 'min': 40.0, 'max': 140.0, 'required': False, 'default': 72.0, 'label': 'Pulse Rate'},
    'RR (breaths/min)': {'type': float, 'min': 10.0, 'max': 40.0, 'required': False, 'default': 18.0, 'label': 'Respiratory Rate'},
    'Cycle(R/I)': {'type': int, 'allowed': [2, 4], 'required': True, 'label': 'Menstrual Cycle Regularity'},
    'Cycle length(days)': {'type': float, 'min': 0.0, 'max': 20.0, 'required': True, 'label': 'Period Flow Length (days)'},
    'Marraige Status (Yrs)': {'type': float, 'min': 0.0, 'max': 50.0, 'required': False, 'default': 0.0, 'label': 'Marriage / Relationship Duration'},
    'Pregnant(Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Current Pregnancy Status'},
    'No. of aborptions': {'type': int, 'min': 0, 'max': 15, 'required': False, 'default': 0, 'label': 'Number of Miscarriages / Abortions'},
    'Hip(inch)': {'type': float, 'min': 20.0, 'max': 70.0, 'required': True, 'label': 'Hip Circumference'},
    'Waist(inch)': {'type': float, 'min': 15.0, 'max': 70.0, 'required': True, 'label': 'Waist Circumference'},
    'Weight gain(Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Weight Gain'},
    'hair growth(Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Excess Hair Growth'},
    'Skin darkening (Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Skin Darkening'},
    'Hair loss(Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Hair Loss'},
    'Pimples(Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Pimples / Acne'},
    'Fast food (Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Fast Food Consumption'},
    'Reg.Exercise(Y/N)': {'type': int, 'allowed': [0, 1], 'required': True, 'label': 'Regular Exercise'}
}


def validate_questionnaire_input(form_data, feature_medians=None):
    """
    Validates submitted form data against FIELD_SPECS.
    
    Returns:
        (sanitized_dict, errors_list)
    """
    sanitized = {}
    errors = []

    for field, spec in FIELD_SPECS.items():
        raw_val = form_data.get(field)

        # Handle empty/missing value
        if raw_val is None or str(raw_val).strip() == '':
            if spec.get('required'):
                errors.append(f"Field '{spec['label']}' is required.")
                continue
            else:
                # Use default or median
                default_val = spec.get('default')
                if default_val is None and feature_medians and field in feature_medians:
                    default_val = feature_medians[field]
                sanitized[field] = spec['type'](default_val)
                continue

        # Type conversion & range validation
        try:
            val = spec['type'](raw_val)
        except (ValueError, TypeError):
            errors.append(f"Field '{spec['label']}' must be a valid number.")
            continue

        # Check allowed discrete set
        if 'allowed' in spec:
            if val not in spec['allowed']:
                errors.append(f"Field '{spec['label']}' has an invalid selection.")
                continue

        # Check min/max bounds
        if 'min' in spec and val < spec['min']:
            errors.append(f"Field '{spec['label']}' cannot be less than {spec['min']}.")
            continue
        if 'max' in spec and val > spec['max']:
            errors.append(f"Field '{spec['label']}' cannot be greater than {spec['max']}.")
            continue

        sanitized[field] = val

    return sanitized, errors
