"""
utils/recommender.py
====================
Rule-based adaptive lifestyle and self-care recommendation engine.
Generates tailored 2-week and 1-month lifestyle plans based on screening risk tier
and individual questionnaire responses (diet, exercise, cycle, BMI).

Adheres strictly to medical safety:
- No medication advice
- No claims of curing, preventing, or treating PCOS
- No promised outcomes
- High-risk tier clearly advises consulting a physician / gynecologist
"""

def generate_recommendation_plans(risk_category, user_data):
    """
    Constructs both 2-week and 1-month structured plans adapted to the patient's inputs.
    
    Returns:
        dict:
            - clinical_advisory: Header advisory tailored to risk category
            - two_week_plan: Dict with 5 pillar sections
            - one_month_plan: Dict with 5 pillar sections
            - tailored_tags: List of personalized tags showing which inputs customized the plan
    """
    cycle_irregular = int(user_data.get('Cycle(R/I)', 2)) in [4, 5]
    fast_food_frequent = int(user_data.get('Fast food (Y/N)', 0)) == 1
    no_regular_exercise = int(user_data.get('Reg.Exercise(Y/N)', 0)) == 0
    weight_gain = int(user_data.get('Weight gain(Y/N)', 0)) == 1
    bmi = float(user_data.get('BMI', 22.0))
    skin_darkening = int(user_data.get('Skin darkening (Y/N)', 0)) == 1

    tailored_tags = []
    if cycle_irregular:
        tailored_tags.append("Cycle Regularity Support")
    if fast_food_frequent:
        tailored_tags.append("Refined Food Reduction & Low-GI Swaps")
    if no_regular_exercise:
        tailored_tags.append("Gentle Activity Ramp-Up")
    else:
        tailored_tags.append("Active Lifestyle Maintenance")
    if bmi >= 25.0 or weight_gain or skin_darkening:
        tailored_tags.append("Insulin Sensitivity & Metabolic Health")

    # 1. Clinical Advisory Header
    if risk_category == "High":
        advisory = (
            "<strong>Important Medical Next Step:</strong> Your questionnaire indicates several prominent risk indicators. "
            "We strongly encourage you to schedule a consultation with a qualified healthcare professional (such as a gynecologist "
            "or reproductive endocrinologist). A clinical diagnosis requires laboratory hormone evaluations (LH, FSH, AMH, androgens) "
            "and pelvic ultrasound imaging that cannot be replaced by this online questionnaire. The lifestyle guidance below "
            "is intended solely to support everyday wellbeing while awaiting your medical consultation."
        )
    elif risk_category == "Moderate":
        advisory = (
            "<strong>Clinical Awareness Advice:</strong> Your responses reflect an intermediate screening profile. "
            "While this does not confirm PCOS, several reported symptoms warrant closer attention. We recommend observing your "
            "cycle and symptoms with the plan below, and seeking professional medical guidance if irregular periods, persistent acne, "
            "or metabolic changes continue."
        )
    else:  # Low
        advisory = (
            "<strong>General Wellness Profile:</strong> Your answers reflect a lower alignment with typical self-reported PCOS indicators. "
            "Maintaining proactive nutrition, restorative sleep, and consistent movement supports long-term reproductive vitality. "
            "If you ever develop cycle unpredictability or new symptoms, always consult a healthcare provider."
        )

    # 2. Nutrition Pillar Adaptation
    if fast_food_frequent or bmi >= 25.0 or skin_darkening:
        two_wk_nutrition = [
            "Focus on low-glycemic index (GI) whole foods: swap refined carbohydrates (white bread, sugary snacks, instant noodles) for complex grains (quinoa, brown rice, oats, legumes).",
            "Pair all meals with healthy fiber and lean protein (eggs, tofu, beans, poultry, fish) to help stabilize post-meal glucose and insulin surges.",
            "Aim to replace fast food meals with simple home-prepared batch options, keeping water or herbal teas as your primary beverages."
        ]
        one_mo_nutrition = [
            "Weeks 1–2: Conduct a pantry audit. Replace ultra-processed packaged foods with whole staples: seeds (flax, chia, pumpkin), leafy greens, berries, and legumes.",
            "Weeks 3–4: Establish balanced plate geometry at meals: 50% non-starchy vegetables, 25% clean protein, 25% slow-burning carbohydrates with a drizzle of healthy fats (olive oil, avocado, nuts).",
            "Target consistent, regular meal intervals rather than prolonged fasting or crash dieting, which can trigger hormonal cortisol spikes."
        ]
    else:
        two_wk_nutrition = [
            "Maintain an anti-inflammatory, nutrient-dense diet emphasizing seasonal vegetables, colorful fruits, lentils, and healthy plant fats.",
            "Stay well hydrated with 2.0–2.5 liters of water daily, reducing sweetened commercial beverages.",
            "Incorporate zinc and antioxidant-rich foods like nuts, pumpkin seeds, and dark berries to support skin vitality."
        ]
        one_mo_nutrition = [
            "Weeks 1–2: Prioritize balanced macro-nutrient distribution across 3 predictable meals, focusing on unprocessed fiber-rich foods.",
            "Weeks 3–4: Explore cyclical meal diversity with prebiotic and probiotic foods (yogurt, fermented foods) to sustain healthy gut microbiome and hormonal metabolism.",
            "Focus on sustained nutritional satisfaction without restrictive calorie deprivation."
        ]

    # 3. Physical Activity Pillar Adaptation
    if no_regular_exercise:
        two_wk_activity = [
            "Begin with low-barrier, gentle movement: aim for a brisk 15–20 minute walk outdoors daily, ideally within 30 minutes after your largest meal.",
            "Incorporate 10 minutes of restorative stretching or beginner yoga before bed to gently activate muscle tone without taxing the adrenal system."
        ]
        one_mo_activity = [
            "Weeks 1–2: Build consistent daily habit of 20–25 minutes of continuous low-impact walking or swimming 4–5 days per week.",
            "Weeks 3–4: Introduce 2 brief weekly bodyweight resistance sessions (15–20 mins: bodyweight squats, wall push-ups, glute bridges) to support skeletal muscle glucose uptake."
        ]
    else:
        two_wk_activity = [
            "Continue your regular exercise program while ensuring you balance high-intensity workouts with adequate muscle recovery.",
            "Incorporate a 10–15 minute leisurely walk after dinner to optimize glucose disposal and encourage evening digestion."
        ]
        one_mo_activity = [
            "Weeks 1–2: Prioritize moderate resistance and strength training 3 times a week, which research highlights as favorable for metabolic hormone balance.",
            "Weeks 3–4: Avoid chronic overtraining or excessive exhausting cardio; integrate at least 2 full rest/mobility days per week to prevent cortisol overproduction."
        ]

    # 4. Sleep Routine Pillar
    two_wk_sleep = [
        "Commit to a consistent 7.5 to 8.5 hour sleep window each night by establishing a fixed wake-up time, including on weekends.",
        "Discontinue all blue-light screens (smartphones, tablets, laptops) at least 45 minutes before turning off lights."
    ]
    one_mo_sleep = [
        "Weeks 1–2: Optimize bedroom environment for restorative sleep (cool room temperature ~18–20°C, dark blackout curtains, minimal ambient noise).",
        "Weeks 3–4: Practice a calming 20-minute wind-down ritual: warm shower, reading, or herbal tea (chamomile/spearmint) to reinforce circadian melatonin production."
    ]

    # 5. Stress Management Pillar
    two_wk_stress = [
        "Practice 5 minutes of boxed breathing (inhale 4s, hold 4s, exhale 4s, hold 4s) twice daily to regulate autonomic sympathetic nervous tone.",
        "Set firm boundaries on evening work communication to guard cognitive downtime."
    ]
    one_mo_stress = [
        "Weeks 1–2: Integrate a daily 10-minute mindfulness or guided somatic relaxation session using a journal or calming audio.",
        "Weeks 3–4: Identify recurring lifestyle stressors and schedule non-negotiable personal recovery windows each week."
    ]

    # 6. Menstrual & Symptom Tracking Pillar
    if cycle_irregular:
        two_wk_tracking = [
            "Begin logging menstrual cycle days in a dedicated tracking app or paper diary: record spotting, flow volume, and cramping.",
            "Note physical and dermatological fluctuations across the next 14 days, including any cystic acne flare-ups or fatigue."
        ]
        one_mo_tracking = [
            "Weeks 1–2: Track basal morning temperature, cervical mucus shifts, and flow dates to build an objective baseline of ovulatory pattern.",
            "Weeks 3–4: Compile your 30-day symptom diary (cycle length, acne, sleep quality, energy levels) into a concise summary ready to share with your physician."
        ]
    else:
        two_wk_tracking = [
            "Log your expected cycle onset date and observe any premenstrual sensations (breast tenderness, mood changes, fluid retention).",
            "Track energy peaks and dips to identify natural monthly hormonal rhythms."
        ]
        one_mo_tracking = [
            "Weeks 1–2: Record standard menstrual flow duration and any intermenstrual spotting.",
            "Weeks 3–4: Maintain consistent monthly cycle records to ensure ongoing stability and early identification of any future cycle changes."
        ]

    return {
        'advisory': advisory,
        'tailored_tags': tailored_tags,
        'two_week_plan': {
            'nutrition': two_wk_nutrition,
            'activity': two_wk_activity,
            'sleep': two_wk_sleep,
            'stress': two_wk_stress,
            'tracking': two_wk_tracking
        },
        'one_month_plan': {
            'nutrition': one_mo_nutrition,
            'activity': one_mo_activity,
            'sleep': one_mo_sleep,
            'stress': one_mo_stress,
            'tracking': one_mo_tracking
        }
    }
