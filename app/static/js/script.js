/* ============================================================
   LUNARA-AI MODERN JAVASCRIPT
   Dark Mode, Form Validation, and Interactive Features
============================================================ */

// ============================================================
// DARK MODE THEME MANAGEMENT
// ============================================================

class ThemeManager {
    constructor() {
        this.currentTheme = this.getTheme();
        this.initTheme();
        this.attachEventListeners();
    }

    getTheme() {
        // Check localStorage first
        const stored = localStorage.getItem('theme');
        if (stored) return stored;
        
        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    }

    initTheme() {
        if (this.currentTheme === 'dark') {
            document.body.classList.add('dark-mode');
            this.updateToggleButton();
        }
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme();
    }

    applyTheme() {
        if (this.currentTheme === 'dark') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        localStorage.setItem('theme', this.currentTheme);
        this.updateToggleButton();
        
        // Send preference to server
        this.sendThemeToServer();
    }

    updateToggleButton() {
        const button = document.querySelector('.theme-toggle');
        if (button) {
            button.innerHTML = this.currentTheme === 'dark' ? '☀️' : '🌙';
            button.title = this.currentTheme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        }
    }

    sendThemeToServer() {
        fetch('/api/theme', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ theme: this.currentTheme })
        }).catch(err => console.log('Theme sync optional'));
    }

    attachEventListeners() {
        const button = document.querySelector('.theme-toggle');
        if (button) {
            button.addEventListener('click', () => this.toggleTheme());
        }
        
        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                if (!localStorage.getItem('theme')) {
                    this.currentTheme = e.matches ? 'dark' : 'light';
                    this.applyTheme();
                }
            });
        }
    }
}

// ============================================================
// FORM VALIDATION
// ============================================================

class FormValidator {
    constructor() {
        this.initValidation();
    }

    initValidation() {
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => this.validateForm(e, form));
        });
    }

    validateForm(e, form) {
        const inputs = form.querySelectorAll('input[type="number"], input[type="text"], select');
        let isValid = true;

        inputs.forEach(input => {
            const name = input.name;
            const value = input.value.trim();

            if (!value && input.required) {
                this.showFieldError(input, 'This field is required');
                isValid = false;
            } else if (input.type === 'number') {
                const numValue = parseFloat(value);
                const validation = this.getValidationRules(name);
                
                if (validation && (numValue < validation.min || numValue > validation.max)) {
                    this.showFieldError(input, `Value must be between ${validation.min} and ${validation.max}`);
                    isValid = false;
                } else {
                    this.clearFieldError(input);
                }
            } else {
                this.clearFieldError(input);
            }
        });

        if (!isValid) {
            e.preventDefault();
        }
    }

    getValidationRules(fieldName) {
        const rules = {
            'age': { min: 15, max: 70 },
            'no_pregnancy': { min: 0, max: 15 },
            'bmi': { min: 15, max: 55 },
            'hdl': { min: 20, max: 100 },
            'systolic_bp': { min: 70, max: 200 },
            'sys_bp': { min: 80, max: 200 },
            'diastolic_bp': { min: 40, max: 120 },
            'dia_bp': { min: 40, max: 120 },
            'ogtt': { min: 50, max: 300 },
            'bs': { min: 4, max: 20 },
            'heart_rate': { min: 40, max: 150 },
        };
        return rules[fieldName];
    }

    showFieldError(input, message) {
        input.classList.add('field-error');
        let errorElement = input.nextElementSibling;
        
        if (!errorElement || !errorElement.classList.contains('error-message')) {
            errorElement = document.createElement('div');
            errorElement.className = 'error-message';
            input.parentNode.insertBefore(errorElement, input.nextSibling);
        }
        
        errorElement.textContent = message;
    }

    clearFieldError(input) {
        input.classList.remove('field-error');
        const errorElement = input.nextElementSibling;
        if (errorElement && errorElement.classList.contains('error-message')) {
            errorElement.remove();
        }
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `flash-message ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="background:none; border:none; cursor:pointer; color:inherit; font-size:1.2rem;">×</button>
    `;
    
    const container = document.querySelector('main');
    if (container) {
        container.insertBefore(notification, container.firstChild);
        setTimeout(() => notification.remove(), 4000);
    }
}

function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString('en-US', options);
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme manager
    new ThemeManager();
    
    // Initialize form validator
    new FormValidator();
    
    // Add smooth active link indicator
    const currentPage = window.location.pathname;
    document.querySelectorAll('nav a').forEach(link => {
        if (link.getAttribute('href') === currentPage) {
            link.style.color = 'var(--color-primary)';
            link.style.borderBottomColor = 'var(--color-primary)';
        }
    });
    
    // Auto-hide flash messages after 5 seconds
    document.querySelectorAll('.flash-message').forEach(msg => {
        setTimeout(() => {
            msg.style.opacity = '0';
            msg.style.transition = 'opacity 0.3s ease';
            setTimeout(() => msg.remove(), 300);
        }, 5000);
    });
});

// ============================================================
// EXPORT FOR BROWSER
// ============================================================

window.ThemeManager = ThemeManager;
window.showNotification = showNotification;
window.formatDate = formatDate;
