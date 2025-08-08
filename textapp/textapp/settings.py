"""
Django settings for textapp project.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv(
    "SECRET_KEY", "django-insecure-nkvdv1#jek-9dfd!d@i75hsvls_bpgd7um1tn48-&%o9d4%jtl"
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Enhanced allowed hosts configuration
allowed_hosts = ["localhost", "127.0.0.1"]
if os.getenv("ALLOWED_HOSTS"):
    allowed_hosts.extend(os.getenv("ALLOWED_HOSTS").split(","))
ALLOWED_HOSTS = allowed_hosts

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "textpage",
    "social_django",  # For Google OAuth
]

if DEBUG:
    INSTALLED_APPS += [
        "debug_toolbar",  # For development debugging
    ]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "social_django.middleware.SocialAuthExceptionMiddleware",  # Add this for social auth
]

if DEBUG:
    MIDDLEWARE.insert(0, "debug_toolbar.middleware.DebugToolbarMiddleware")

ROOT_URLCONF = "textapp.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "social_django.context_processors.backends",  # Required for social auth
                "social_django.context_processors.login_redirect",
                # Removed the problematic line: "textpage.context_processors.ai_available"
            ],
        },
    },
]

WSGI_APPLICATION = "textapp.wsgi.application"

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Authentication Backends
AUTHENTICATION_BACKENDS = (
    "social_core.backends.google.GoogleOAuth2",
    "django.contrib.auth.backends.ModelBackend",
)

# Google OAuth2 credentials
SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = os.getenv("GOOGLE_OAUTH2_KEY")
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = os.getenv("GOOGLE_OAUTH2_SECRET")

# Social Auth Settings
SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = ["openid", "email", "profile"]

SOCIAL_AUTH_GOOGLE_OAUTH2_USE_DEPRECATED_API = False
SOCIAL_AUTH_ADMIN_USER_SEARCH_FIELDS = ["username", "first_name", "email"]

# Login / Redirect URLs
LOGIN_URL = "/auth/login/"  # URL for login page
LOGIN_REDIRECT_URL = "/home/"  # Redirect after successful login
LOGOUT_REDIRECT_URL = "/auth/login/"  # Redirect after logout

# Social auth redirect URLs
SOCIAL_AUTH_LOGIN_REDIRECT_URL = "/home/"
SOCIAL_AUTH_LOGIN_ERROR_URL = "/auth/login/"
SOCIAL_AUTH_NEW_USER_REDIRECT_URL = "/home/"
SOCIAL_AUTH_NEW_ASSOCIATION_REDIRECT_URL = "/home/"

# Social auth pipeline settings
SOCIAL_AUTH_PIPELINE = (
    "social_core.pipeline.social_auth.social_details",
    "social_core.pipeline.social_auth.social_uid",
    "social_core.pipeline.social_auth.auth_allowed",
    "social_core.pipeline.social_auth.social_user",
    "social_core.pipeline.user.get_username",
    "social_core.pipeline.user.create_user",
    "social_core.pipeline.social_auth.associate_user",
    "social_core.pipeline.social_auth.load_extra_data",
    "social_core.pipeline.user.user_details",
)

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]  # Add this if you have a static directory

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# AI Model Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DIALOGPT_AVAILABLE = False  # Add this for context processors

# Internal IPs for Debug Toolbar
if DEBUG:
    INTERNAL_IPS = ["127.0.0.1"]

# Enhanced logging configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose" if DEBUG else "simple",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "debug.log",
            "formatter": "verbose",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
        "textpage": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "google.generativeai": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    },
}
