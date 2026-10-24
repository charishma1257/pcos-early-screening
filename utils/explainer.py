"""
utils/explainer.py
==================
XAI explainability using SHAP TreeExplainer for XGBoost.
Computes patient-level feature contributions, formats plain-English explanations,
and renders a visually distinct, spacious horizontal bar chart as a base64 string.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap


def format_user_value(feature_name, raw_val):
    """Formats raw questionnaire input into patient-friendly readable text."""
    if feature_name == 'Cycle(R/I)':
        return 'Irregular cycle' if raw_val in [4, 5] else 'Regular cycle'
    if feature_name in ['Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
                        'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'Pregnant(Y/N)']:
        return 'Yes' if int(raw_val) == 1 else 'No'
    if feature_name == 'Age (yrs)':
        return f"{int(raw_val)} yrs"
    if feature_name == 'BMI':
        return f"{raw_val:.1f} kg/m²"
    if feature_name == 'Cycle length(days)':
        return f"{int(raw_val)} days flow"
    if feature_name in ['Waist(inch)', 'Hip(inch)']:
        return f"{raw_val:.1f} in"
    if feature_name == 'Pulse rate(bpm)':
        return f"{int(raw_val)} bpm"
    if feature_name == 'RR (breaths/min)':
        return f"{int(raw_val)} bpm"
    if feature_name == 'Marraige Status (Yrs)':
        return f"{int(raw_val)} yrs"
    if feature_name == 'No. of aborptions':
        return f"{int(raw_val)}"
    return str(raw_val)


def compute_shap_explanation(base_xgb_model, preprocessor, feature_order, display_names, user_dict, top_k=5):
    """
    Computes SHAP values for the single patient input vector.
    
    Returns:
        dict containing:
            - top_factors: list of dicts (feature, display_name, user_val_str, shap_val, direction)
            - narrative: structured plain-English narrative explanation with clear paragraph spacing
            - chart_base64: base64-encoded PNG image of the top contributors
    """
    # 1. Convert user dict to single-row DataFrame with exact feature order
    input_df = pd.DataFrame([user_dict])[feature_order]

    # 2. Transform through the preprocessor
    transformed_input = preprocessor.transform(input_df)

    # 3. Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(base_xgb_model)
    shap_results = explainer(transformed_input)

    # Values for this patient
    shap_vals = shap_results.values[0]

    # 4. Pair features with their SHAP values and user inputs
    contributions = []
    for idx, col in enumerate(feature_order):
        s_val = float(shap_vals[idx])
        raw_val = user_dict.get(col)
        contributions.append({
            'feature': col,
            'display_name': display_names.get(col, col),
            'raw_value': raw_val,
            'user_val_str': format_user_value(col, raw_val),
            'shap_value': s_val,
            'abs_shap': abs(s_val),
            'direction': 'increases_risk' if s_val > 0 else 'decreases_risk'
        })

    # Sort by absolute magnitude of contribution
    contributions_sorted = sorted(contributions, key=lambda x: x['abs_shap'], reverse=True)
    top_factors = contributions_sorted[:top_k]

    # 5. Build dynamic plain-English narrative with clear paragraph structure and spacing
    pos_factors = [f for f in top_factors if f['shap_value'] > 0]
    neg_factors = [f for f in top_factors if f['shap_value'] < 0]

    paragraphs = []

    if pos_factors:
        pos_descriptions = [
            f"<strong>{f['display_name']}</strong> ({f['user_val_str']})"
            for f in pos_factors
        ]
        if len(pos_descriptions) == 1:
            pos_joined = pos_descriptions[0]
        elif len(pos_descriptions) == 2:
            pos_joined = f"{pos_descriptions[0]} and {pos_descriptions[1]}"
        else:
            pos_joined = ", ".join(pos_descriptions[:-1]) + f", and {pos_descriptions[-1]}"

        paragraphs.append(
            f"<p class=\"mb-3 leading-relaxed\">The primary factors that <strong>contributed positively toward raising the model's estimate</strong> were your reported {pos_joined}.</p>"
        )

    if neg_factors:
        neg_descriptions = [
            f"<strong>{f['display_name']}</strong> ({f['user_val_str']})"
            for f in neg_factors
        ]
        if len(neg_descriptions) == 1:
            neg_joined = neg_descriptions[0]
        elif len(neg_descriptions) == 2:
            neg_joined = f"{neg_descriptions[0]} and {neg_descriptions[1]}"
        else:
            neg_joined = ", ".join(neg_descriptions[:-1]) + f", and {neg_descriptions[-1]}"

        paragraphs.append(
            f"<p class=\"mb-3 leading-relaxed\">Conversely, reported factors that <strong>contributed toward lowering the model's estimate</strong> were your {neg_joined}.</p>"
        )

    paragraphs.append(
        "<p class=\"text-muted small mb-0 pt-2 border-top border-lavender leading-relaxed\">"
        "<em>Medical Note:</em> These factors explain how this machine-learning algorithm weighted your questionnaire responses. "
        "SHAP values reflect statistical model weights, not biological causes or clinical certainty."
        "</p>"
    )

    full_narrative = "".join(paragraphs)

    # 6. Render Modern, Clean Horizontal Bar Chart Server-Side
    # Display top factors inverted so the highest impact feature is at the top
    chart_factors = top_factors[::-1]
    
    # Clean two-line label: Feature Display Name on top, User Input cleanly underneath
    labels = [f"{f['display_name']}\n(Input: {f['user_val_str']})" for f in chart_factors]
    values = [f['shap_value'] for f in chart_factors]

    # Harmonious colors matching application palette
    # Positive (increases risk): Deep Rose/Crimson #c83264
    # Negative (decreases risk): Calm Green/Teal #198754
    bar_colors = ['#c83264' if v > 0 else '#198754' for v in values]

    # Spacious figure sizing for crisp readability on desktop and mobile
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Y-positions with generous vertical spacing
    y_positions = np.arange(len(chart_factors))
    bar_height = 0.46
    bars = ax.barh(y_positions, values, color=bar_colors, height=bar_height, edgecolor='none', zorder=3)

    # Zero reference line
    ax.axvline(0, color='#6c757d', linestyle='-', linewidth=1.2, alpha=0.85, zorder=2)

    # Subtle horizontal/vertical grid
    ax.grid(axis='x', color='#edf2f7', linestyle='--', linewidth=0.8, alpha=0.9, zorder=1)

    # Y-ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight='500', color='#212529', linespacing=1.25)
    ax.tick_params(axis='y', which='both', length=0, pad=12)

    # X-axis limits: generous symmetric headroom so end annotations never clip
    max_abs_val = max(abs(min(values)), abs(max(values)))
    x_limit = max(max_abs_val * 1.45, 0.5)
    ax.set_xlim(-x_limit, x_limit)

    ax.set_xlabel('SHAP Feature Impact on Risk Estimate', fontsize=9.5, fontweight='600', color='#495057', labelpad=10)
    ax.tick_params(axis='x', labelsize=8.5, colors='#6c757d')

    # Clean borders (remove top, right, and bottom spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ced4da')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#ced4da')
    ax.spines['bottom'].set_linewidth(0.8)

    # Neat value annotations placed right at the ends of the bars
    offset_val = x_limit * 0.035
    for bar, val in zip(bars, values):
        width = bar.get_width()
        if width >= 0:
            ha = 'left'
            x_text = width + offset_val
            val_str = f"+{width:.2f}"
            text_color = '#a82050'
        else:
            ha = 'right'
            x_text = width - offset_val
            val_str = f"{width:.2f}"
            text_color = '#0f6840'

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            val_str,
            va='center',
            ha=ha,
            fontsize=9.5,
            fontweight='700',
            color=text_color,
            zorder=4
        )

    # Clear legend placed at top outside the plot area (never collides with bars!)
    legend_elements = [
        Patch(facecolor='#c83264', label='Pushes risk estimate higher (+)'),
        Patch(facecolor='#198754', label='Pushes risk estimate lower (-)')
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=True,
        facecolor='#f8f9fa',
        edgecolor='#dee2e6',
        fontsize=9,
        handlelength=1.2,
        handleheight=0.8,
        borderpad=0.5
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'top_factors': top_factors,
        'narrative': full_narrative,
        'chart_base64': chart_base64
    }
