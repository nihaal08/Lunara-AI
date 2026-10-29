"""Tests for LUNARA-AI web application routes and authentication."""

import unittest
from app import create_app
from app.config import TestingConfig


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def test_home_page(self):
        """Test that home page renders successfully."""
        response = self.client.get("/", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"LUNARA", response.data)

    def test_about_page(self):
        """Test that about page renders successfully."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About", response.data)

    def test_contact_page(self):
        """Test that contact page renders successfully."""
        response = self.client.get("/contact")
        self.assertEqual(response.status_code, 200)

    def test_login_page(self):
        """Test that login page renders successfully."""
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Login", response.data)

    def test_signup_page(self):
        """Test that signup page renders successfully."""
        response = self.client.get("/signup")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Sign Up", response.data)

    def test_prediction_auth_redirect(self):
        """Test that unauthenticated prediction access redirects to login."""
        response = self.client.get("/predict/gdm")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))


if __name__ == "__main__":
    unittest.main()
