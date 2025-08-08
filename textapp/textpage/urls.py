from django.urls import path, include
from django.http import HttpResponse
from django.views.generic import RedirectView, TemplateView
from django.contrib.auth import views as auth_views
from django.conf import settings
from . import views


def favicon_view(request):
    return HttpResponse(status=204)


def robots_txt(request):
    lines = [
        "User-Agent: *",
        "Disallow: /admin/",
        "Disallow: /api/",
        "Allow: /",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")


# API URLs for AJAX endpoints
api_patterns = [
    path("send/", views.send, name="send_message"),
    path("messages/<str:room>/", views.getMessages, name="get_messages"),
    path("ai-response/", views.get_ai_response, name="get_ai_response"),
]

# Authentication URLs
auth_patterns = [
    path(
        "login/",
        TemplateView.as_view(template_name="LogInPage.html"),
        name="login",
    ),
    path(
        "logout/",
        auth_views.LogoutView.as_view(next_page="/auth/login/"),
        name="logout",
    ),
]

# AI Chat URLs
ai_patterns = [
    path("", views.ai_home, name="ai_home"),
    path("chat/", views.ai_chat, name="ai_chat"),
    path("chat/<str:model>/", views.ai_chat_with_model, name="ai_chat_with_model"),
    path("test-gemini/", views.test_gemini_connection, name="test_gemini"),
]

# Main URL patterns
urlpatterns = [
    path("test-room-model/", views.test_room_model, name="test_room_model"),
    path("favicon.ico", favicon_view, name="favicon"),
    path("robots.txt", robots_txt, name="robots_txt"),
    # Root redirect
    path("", views.root_redirect, name="root_redirect"),
    # Home page (requires login)
    path("home/", views.home, name="home"),
    # Other paths
    path("checkview/", views.checkview, name="checkview"),
    path("room/<str:room>/", views.room, name="room"),
    # Include all URL groups
    path("api/", include((api_patterns, "api"))),
    path("ai/", include((ai_patterns, "ai"))),
    path("auth/", include((auth_patterns, "auth"))),
    # Social auth URLs
    path("social-auth/", include("social_django.urls", namespace="social")),
]

# Add debug toolbar in development
if settings.DEBUG:
    try:
        import debug_toolbar

        urlpatterns = [
            path("__debug__/", include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass  # debug_toolbar not installed
