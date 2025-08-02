document.addEventListener('DOMContentLoaded', () => {
    // Form validation for all forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', (e) => {
            let valid = true;
            const inputs = form.querySelectorAll('input, select, textarea');

            inputs.forEach(input => {
                const name = input.name;
                const value = input.value;
                let error = null;

                // Validation rules based on Streamlit ranges
                if (input.type === 'number') {
                    const numValue = parseFloat(value);
                    if (name === 'age') {
                        if (form.id === 'gdm_form' || form.id === 'preeclampsia_form') {
                            if (numValue < 18 || numValue > 50) {
                                error = 'Age must be between 18 and 50.';
                            }
                        } else if (form.id === 'maternal_form') {
                            if (numValue < 10 || numValue > 70) {
                                error = 'Age must be between 10 and 70.';
                            }
                        }
                    } else if (name === 'no_pregnancy' || name === 'gravida' || name === 'parity') {
                        if (numValue < 0 || numValue > 10) {
                            error = `${input.name} must be between 0 and 10.`;
                        }
                    } else if (name === 'bmi') {
                        if (numValue < 15.0 || numValue > 50.0) {
                            error = 'BMI must be between 15.0 and 50.0.';
                        }
                    } else if (name === 'hdl') {
                        if (numValue < 20.0 || numValue > 100.0) {
                            error = 'HDL must be between 20.0 and 100.0.';
                        }
                    } else if (name === 'sys_bp' || name === 'systolic_bp') {
                        if (numValue < (name === 'sys_bp' ? 80.0 : 70.0) || numValue > 200.0) {
                            error = `${input.name} must be between ${name === 'sys_bp' ? '80.0' : '70.0'} and 200.0.`;
                        }
                    } else if (name === 'dia_bp' || name === 'diastolic_bp') {
                        if (numValue < 40.0 || numValue > 120.0) {
                            error = `${input.name} must be between 40.0 and 120.0.`;
                        }
                    } else if (name === 'ogtt') {
                        if (numValue < 50.0 || numValue > 300.0) {
                            error = 'OGTT must be between 50.0 and 300.0.';
                        }
                    } else if (name === 'bs') {
                        if (numValue < 4.0 || numValue > 20.0) {
                            error = 'Blood Sugar must be between 4.0 and 20.0.';
                        }
                    } else if (name === 'heart_rate') {
                        if (numValue < 40 || numValue > 120) {
                            error = 'Heart Rate must be between 40 and 120.';
                        }
                    } else if (name === 'gestational_age') {
                        if (numValue < 10.0 || numValue > 42.0) {
                            error = 'Gestational Age must be between 10.0 and 42.0.';
                        }
                    } else if (name === 'hb') {
                        if (numValue < 5.0 || numValue > 18.0) {
                            error = 'Hemoglobin must be between 5.0 and 18.0.';
                        }
                    } else if (name === 'fetal_weight') {
                        if (numValue < 0.1 || numValue > 5.0) {
                            error = 'Fetal Weight must be between 0.1 and 5.0.';
                        }
                    } else if (name === 'amniotic_fluid') {
                        if (numValue < 5.0 || numValue > 25.0) {
                            error = 'Amniotic Fluid Levels must be between 5.0 and 25.0.';
                        }
                    }
                } else if (input.type === 'select-one') {
                    const intValue = parseInt(value);
                    if (intValue !== 0 && intValue !== 1) {
                        error = `${input.name} must be 0 (No) or 1 (Yes).`;
                    }
                }

                if (error) {
                    valid = false;
                    input.style.borderColor = '#721C24';
                    alert(error);
                } else {
                    input.style.borderColor = '#E5E5E5';
                }
            });

            if (!valid) {
                e.preventDefault();
            }
        });
    });

    // Admin dashboard charts
    if (document.getElementById('predictionChart')) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        const modelCounts = JSON.parse(document.getElementById('modelCounts').textContent);
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: modelCounts.map(item => item[0]),
                datasets: [{
                    label: 'Prediction Count',
                    data: modelCounts.map(item => item[1]),
                    backgroundColor: '#F5A7A6',
                    borderColor: '#E59695',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#333333' }
                    },
                    x: {
                        ticks: { color: '#333333' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#333333' } }
                }
            }
        });
    }
});