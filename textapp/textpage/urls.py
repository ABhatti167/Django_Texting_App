from django.urls import path
from django.http import HttpResponse
from . import views


def favicon_view(request):
    return HttpResponse(status=204)


urlpatterns = [
    path("", views.home, name="home"),
    path("favicon.ico", favicon_view),
    path("checkview/", views.checkview, name="checkview"),
    path("send/", views.send, name="send"),
    path("getMessages/<str:room>/", views.getMessages, name="getmessages"),
    # AI Chat URLs
    path("ai-home/", views.ai_home, name="ai_home"),
    path("ai-chat/", views.ai_chat, name="ai_chat"),
    path("ai-chat/<str:model>/", views.ai_chat_with_model, name="ai_chat_with_model"),
    path("ai-response/", views.get_ai_response, name="get_ai_response"),
    # Move the room pattern to the end and prefix it with /room/
    path("room/<str:room>/", views.room, name="room"),
]
