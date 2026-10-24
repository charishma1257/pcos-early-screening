/**
 * main.js
 * Client-side validation and interactive helpers for PCOS Screening Prototype.
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. BMI Quick Calculator Helper
    const quickHeight = document.getElementById('quickHeight');
    const quickWeight = document.getElementById('quickWeight');
    const btnApplyBmi = document.getElementById('btnApplyBmi');
    const bmiInput = document.getElementById('bmi');

    if (btnApplyBmi && quickHeight && quickWeight && bmiInput) {
        btnApplyBmi.addEventListener('click', function () {
            const h = parseFloat(quickHeight.value);
            const w = parseFloat(quickWeight.value);

            if (isNaN(h) || h < 100 || h > 220) {
                alert('Please enter a valid height between 100 and 220 cm.');
                quickHeight.focus();
                return;
            }

            if (isNaN(w) || w < 30 || w > 200) {
                alert('Please enter a valid weight between 30 and 200 kg.');
                quickWeight.focus();
                return;
            }

            const bmiVal = w / ((h / 100) * (h / 100));
            bmiInput.value = bmiVal.toFixed(1);

            // Collapse calculator panel
            const collapseEl = document.getElementById('bmiCalcCollapse');
            if (collapseEl && window.bootstrap) {
                const bsCollapse = bootstrap.Collapse.getInstance(collapseEl) || new bootstrap.Collapse(collapseEl);
                bsCollapse.hide();
            }
        });
    }

    // 2. Client-side Form Validation
    const screeningForm = document.getElementById('screeningForm');
    if (screeningForm) {
        screeningForm.addEventListener('submit', function (event) {
            let isValid = true;
            const requiredInputs = screeningForm.querySelectorAll('input[required]');

            requiredInputs.forEach(function (input) {
                if (input.type === 'radio') {
                    const name = input.name;
                    const checked = screeningForm.querySelector(`input[name="${name}"]:checked`);
                    if (!checked) {
                        isValid = false;
                        input.classList.add('is-invalid');
                    } else {
                        input.classList.remove('is-invalid');
                    }
                } else {
                    if (!input.value.trim() || input.value === '') {
                        isValid = false;
                        input.classList.add('is-invalid');
                    } else {
                        input.classList.remove('is-invalid');
                    }
                }
            });

            if (!isValid) {
                event.preventDefault();
                event.stopPropagation();
                alert('Please fill in all required questionnaire fields before submitting.');
            }
        });
    }
});
