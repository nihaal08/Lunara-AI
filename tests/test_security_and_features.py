"""Comprehensive security, functionality, and edge-case test suite for LUNARA-AI."""

import unittest
from app import create_app
from app.config import Config, TestingConfig
from app.services.pdf_service import generate_prediction_pdf


class SecurityConfig(TestingConfig):
    """Configuration for testing security mechanisms like CSRF."""
    TESTING = False
    WTF_CSRF_ENABLED = True


class SecurityAndFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def test_security_headers_present(self):
        """Test that defensive HTTP security headers are enforced."""
        response = self.client.get("/")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(response.headers.get("X-XSS-Protection"), "1; mode=block")
        self.assertIn("Strict-Transport-Security", response.headers)
        self.assertIn("Content-Security-Policy", response.headers)

    def test_open_redirect_protection(self):
        """Test that malicious external redirect URLs are blocked upon login."""
        # 1. Register a test user
        self.client.post(
            "/signup",
            data={
                "username": "redirect_user",
                "full_name": "Redirect Test",
                "email": "redirect@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )

        # 2. Attempt login with malicious open redirect
        response = self.client.post(
            "/login?next=https://evil-phishing-site.com",
            data={"identifier": "redirect_user", "password": "Password123!"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        # Must redirect safely to internal home, NOT to external evil domain
        location = response.headers.get("Location", "")
        self.assertNotIn("evil-phishing-site.com", location)
        self.assertTrue(location.endswith("/home") or location == "/home")

    def test_safe_internal_redirect(self):
        """Test that safe internal next paths are honored upon login."""
        self.client.post(
            "/signup",
            data={
                "username": "safe_user",
                "full_name": "Safe User",
                "email": "safe@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )
        response = self.client.post(
            "/login?next=/predict/maternal",
            data={"identifier": "safe_user", "password": "Password123!"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        location = response.headers.get("Location", "")
        self.assertTrue(location.endswith("/predict/maternal"))

    def test_user_authentication_flow(self):
        """Test complete signup, authentication, and logout cycle."""
        # Register user
        res_signup = self.client.post(
            "/signup",
            data={
                "username": "testpatient",
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "password": "SecurePassword1!",
                "confirm_password": "SecurePassword1!",
            },
            follow_redirects=True,
        )
        self.assertEqual(res_signup.status_code, 200)

        # Duplicate username should be rejected
        res_dup = self.client.post(
            "/signup",
            data={
                "username": "testpatient",
                "full_name": "Another Jane",
                "email": "jane2@example.com",
                "password": "SecurePassword1!",
                "confirm_password": "SecurePassword1!",
            },
            follow_redirects=True,
        )
        self.assertIn(b"already exists", res_dup.data)

        # Wrong password rejected
        res_bad_pw = self.client.post(
            "/login",
            data={"identifier": "testpatient", "password": "WrongPassword!"},
            follow_redirects=True,
        )
        self.assertIn(b"Invalid username", res_bad_pw.data)

        # Correct password succeeds
        res_login = self.client.post(
            "/login",
            data={"identifier": "testpatient", "password": "SecurePassword1!"},
            follow_redirects=True,
        )
        self.assertEqual(res_login.status_code, 200)
        self.assertIn(b"Welcome back, testpatient!", res_login.data)

        # Logout clears session
        res_logout = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(res_logout.status_code, 200)
        self.assertIn(b"logged out", res_logout.data)

    def test_admin_authorization(self):
        """Test admin route access control for normal users and admin users."""
        # Non-admin user
        self.client.post(
            "/signup",
            data={
                "username": "patient_only",
                "full_name": "Normal Patient",
                "email": "patient@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )
        self.client.post(
            "/login",
            data={"identifier": "patient_only", "password": "Password123!"},
        )

        res_unauthorized = self.client.get("/admin/dashboard")
        self.assertEqual(res_unauthorized.status_code, 403)
        self.assertIn(b"Access Denied", res_unauthorized.data)

        # Direct /admin route also protected
        res_admin_root = self.client.get("/admin")
        self.assertEqual(res_admin_root.status_code, 403)

    def test_prediction_and_pdf_generation(self):
        """Test running a prediction, storing history, and exporting PDF."""
        # Authenticate user
        self.client.post(
            "/signup",
            data={
                "username": "prediction_user",
                "full_name": "Maternal Patient",
                "email": "maternal@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )
        self.client.post(
            "/login",
            data={"identifier": "prediction_user", "password": "Password123!"},
        )

        # Run GDM prediction
        gdm_post = self.client.post(
            "/predict/gdm",
            data={
                "age": 32, "no_pregnancy": 2, "bmi": 26.5, "hdl": 52,
                "family_history": 0, "large_child": 0, "pcos": 0,
                "sys_bp": 118, "dia_bp": 78, "ogtt": 110,
                "sedentary": 0, "prediabetes": 0,
            },
            follow_redirects=True,
        )
        self.assertEqual(gdm_post.status_code, 200)
        self.assertIn(b"Prediction completed successfully!", gdm_post.data)

        # Check history view
        res_history = self.client.get("/history")
        self.assertEqual(res_history.status_code, 200)
        self.assertIn(b"Gestational Diabetes", res_history.data)

        # Download history PDF
        res_pdf = self.client.post("/download-history")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.mimetype, "application/pdf")
        self.assertTrue(len(res_pdf.data) > 500)
        self.assertTrue(res_pdf.data.startswith(b"%PDF"))

        # Clear history
        res_clear = self.client.post("/clear-history", follow_redirects=True)
        self.assertEqual(res_clear.status_code, 200)
        self.assertIn(b"cleared successfully", res_clear.data)

    def test_prediction_input_bounds_validation(self):
        """Test that out-of-bounds inputs are caught and error messages displayed."""
        self.client.post(
            "/signup",
            data={
                "username": "bounds_user",
                "full_name": "Bounds Tester",
                "email": "bounds@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )
        self.client.post(
            "/login",
            data={"identifier": "bounds_user", "password": "Password123!"},
        )

        # Invalid Age (> 65)
        res_invalid_age = self.client.post(
            "/predict/gdm",
            data={"age": 99, "no_pregnancy": 1, "bmi": 24, "hdl": 50, "family_history": 0,
                  "large_child": 0, "pcos": 0, "sys_bp": 120, "dia_bp": 80, "ogtt": 100,
                  "sedentary": 0, "prediabetes": 0},
            follow_redirects=True,
        )
        self.assertIn(b"Age must be between 15 and 65", res_invalid_age.data)

        # Invalid Maternal BP (> 220)
        res_invalid_bp = self.client.post(
            "/predict/maternal",
            data={"age": 25, "systolic_bp": 350, "diastolic_bp": 80, "bs": 7.0, "heart_rate": 70},
            follow_redirects=True,
        )
        self.assertIn(b"Systolic BP must be between 60 and 220", res_invalid_bp.data)

    def test_theme_api(self):
        """Test API theme preference setting."""
        response = self.client.post("/api/theme", json={"theme": "dark"})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data.get("theme"), "dark")
        self.assertIn("theme=dark", response.headers.get("Set-Cookie", ""))


if __name__ == "__main__":
    unittest.main()
