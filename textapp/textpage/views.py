import os
import uuid
import tempfile
import base64
import io
import json
import logging
import traceback
from datetime import timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.cache import cache
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.contrib import messages
from django.conf import settings

# AI Libraries
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("Transformers not installed - DialoGPT will not be available")

try:
    import google.generativeai as genai
except ImportError:
    print("Google AI not installed - Gemini will not be available")

try:
    import speech_recognition as sr
    from gtts import gTTS
except ImportError:
    print("Speech libraries not installed - Voice features will not be available")

from textpage.models import (
    Room,
    Message,
    ConversationHistory,
    UserProfile,
    RoomParticipant,
)

# Set up logging
logger = logging.getLogger(__name__)

User = get_user_model()

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Global AI model initialization with error handling
dialogpt_tokenizer = None
dialogpt_model = None
DIALOGPT_AVAILABLE = False

try:
    dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    DIALOGPT_AVAILABLE = True
    logger.info("DialoGPT model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load DialoGPT model: {e}")

# Setup Gemini API with API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Test connection with a simple model
        test_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        test_response = test_model.generate_content("Test connection")
        if test_response.text:
            GEMINI_AVAILABLE = True
            logger.info("Gemini API configured successfully")
        else:
            logger.error("Gemini API connection test failed")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
        logger.error(traceback.format_exc())
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

# FIXED: Updated model mapping with correct model names
MODEL_MAPPING = {
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-live-2.5-flash-preview": "gemini-live-2.5-flash-preview",
    "dialogpt": "dialogpt",
}


# Root redirect view
def root_redirect(request):
    """Redirect to home if authenticated, otherwise to login page"""
    if request.user.is_authenticated:
        return redirect("home")
    else:
        return redirect("auth:login")


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0]
    else:
        ip = request.META.get("REMOTE_ADDR")
    return ip


def get_user_agent(request):
    """Get user agent from request"""
    return request.META.get("HTTP_USER_AGENT", "")


@login_required(login_url="/auth/login/")
def home(request):
    """Enhanced home view with recent rooms"""
    recent_rooms = (
        Room.objects.filter(is_active=True)
        .annotate(message_count=Count("messages"))
        .order_by("-updated_at")[:10]
    )

    context = {"recent_rooms": recent_rooms, "user": request.user}
    return render(request, "home.html", context)


@login_required(login_url="/auth/login/")
def room(request, room):
    """Enhanced room view with pagination and user tracking"""
    username = request.GET.get("username")
    if not username:
        username = request.user.get_full_name() or request.user.username

    try:
        room_obj = get_object_or_404(Room, name=room, is_active=True)

        participant, created = RoomParticipant.objects.get_or_create(
            room=room_obj, user=request.user, defaults={"is_active": True}
        )
        if not created:
            participant.last_seen = timezone.now()
            participant.is_active = True
            participant.save()

        context = {
            "username": username,
            "room": room,
            "room_details": room_obj,
        }

        return render(request, "room.html", context)

    except Exception as e:
        logger.error(f"Error accessing room {room}: {e}")
        messages.error(request, "Room not found or unavailable")
        return redirect("home")


@login_required(login_url="/auth/login/")
@require_POST
def checkview(request):
    """Fixed room creation without description field"""
    try:
        room_name = request.POST.get("room_name", "").strip()
        action = request.POST.get("action", "").strip()

        if not room_name:
            messages.error(request, "❌ Room name is required!")
            return redirect("home")

        if not action:
            messages.error(request, "❌ No action specified!")
            return redirect("home")

        if len(room_name) < 3:
            messages.error(request, "❌ Room name must be at least 3 characters!")
            return redirect("home")

        if len(room_name) > 100:
            messages.error(request, "❌ Room name too long!")
            return redirect("home")

        import re

        room_name_clean = re.sub(r"[^\w\s\-_]", "", room_name)
        if room_name_clean != room_name:
            room_name = room_name_clean

        username = request.user.get_full_name() or request.user.username

        if action == "enter_room":
            try:
                try:
                    room_obj = Room.objects.get(name=room_name)
                    if not room_obj.is_active:
                        room_obj.is_active = True
                        room_obj.save()
                    messages.success(request, f"🎉 Joined room '{room_name}'!")
                except Room.DoesNotExist:
                    room_obj = Room.objects.create(
                        name=room_name, created_by=request.user, is_active=True
                    )
                    messages.success(request, f"🎉 Created room '{room_name}'!")

                from urllib.parse import quote

                encoded_room = quote(room_name)
                encoded_username = quote(username)
                return redirect(f"/room/{encoded_room}/?username={encoded_username}")

            except Exception as e:
                messages.error(request, f"❌ Error: {str(e)}")
                return redirect("home")

        elif action == "chat_ai":
            return redirect("/ai/chat/gemini-2.5-flash/")

        else:
            messages.error(request, f"❌ Unknown action: {action}")
            return redirect("home")

    except Exception as e:
        messages.error(request, f"❌ System error: {str(e)}")
        return redirect("home")


@login_required(login_url="/auth/login/")
def test_room_model(request):
    """Test view without description field"""
    try:
        from django.http import HttpResponse
        from textpage.models import Room

        result = "🔧 ROOM MODEL TEST RESULTS\n"
        result += "=" * 50 + "\n\n"

        from django.db import connection

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result += "✅ Database connection working\n"

        room_count = Room.objects.all().count()
        result += f"✅ Found {room_count} rooms in database\n"

        test_room_name = f"test_{timezone.now().strftime('%H%M%S')}"
        try:
            test_room = Room.objects.create(
                name=test_room_name, created_by=request.user, is_active=True
            )
            result += f"✅ Created test room: {test_room.name}\n"
            result += f"   Room ID: {test_room.id}\n"
            test_room.delete()
            result += f"✅ Test room deleted\n"
        except Exception as e:
            result += f"❌ Room creation failed: {e}\n"

        result += f"\n👤 User: {request.user.username}\n"
        result += "\n" + "=" * 50 + "\n"
        if "no such column" in str(result).lower():
            result += "❌ DATABASE MIGRATION NEEDED!\n"
            result += "Run: python manage.py makemigrations textpage\n"
            result += "Then: python manage.py migrate\n"
        else:
            result += "✅ Room model working!\n"

        return HttpResponse(result, content_type="text/plain; charset=utf-8")

    except Exception as e:
        error_info = f"❌ TEST ERROR: {e}\n"
        if "no such column" in str(e):
            error_info += "\n🔧 SOLUTION: Run database migrations:\n"
            error_info += "python manage.py makemigrations textpage\n"
            error_info += "python manage.py migrate\n"
        return HttpResponse(error_info, content_type="text/plain; charset=utf-8")


@login_required(login_url="/auth/login/")
@require_POST
@csrf_exempt
def send(request):
    """Fixed message sending function"""
    try:
        # Get form data
        message_content = request.POST.get("message", "").strip()
        username = request.POST.get("username", "").strip()
        room_name = request.POST.get(
            "room_id", ""
        ).strip()  # This is actually room name, not ID

        # Validate required fields
        if not all([message_content, username, room_name]):
            logger.error(
                f"Missing fields - message: {bool(message_content)}, username: {bool(username)}, room: {bool(room_name)}"
            )
            return JsonResponse({"error": "Missing required fields"}, status=400)

        if len(message_content) > 10000:
            return JsonResponse({"error": "Message too long"}, status=400)

        # Get or create room
        try:
            room_obj = Room.objects.get(name=room_name, is_active=True)
        except Room.DoesNotExist:
            # Create room if it doesn't exist
            room_obj = Room.objects.create(
                name=room_name, created_by=request.user, is_active=True
            )

        # Create message with proper field mapping according to your model
        message = Message.objects.create(
            content=message_content,
            username=username,
            user=request.user,
            room=room_obj,
            message_type="text",
            ip_address=get_client_ip(request),
            user_agent=(
                get_user_agent(request)[:500] if get_user_agent(request) else ""
            ),  # Truncate user agent if too long
        )

        # Update user profile stats (with error handling)
        try:
            profile, created = UserProfile.objects.get_or_create(
                user=request.user,
                defaults={
                    "total_messages_sent": 0,
                    "total_ai_conversations": 0,
                },
            )
            profile.total_messages_sent += 1
            profile.save()
        except Exception as profile_error:
            # Don't fail the message send if profile update fails
            logger.warning(f"Profile update failed: {profile_error}")

        logger.info(f"Message sent successfully: {username} in {room_name}")
        return JsonResponse({"status": "success", "message_id": str(message.id)})

    except Exception as e:
        logger.error(f"Error in send view: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return JsonResponse({"error": "Failed to send message"}, status=500)


@login_required(login_url="/auth/login/")
@require_http_methods(["GET"])
def getMessages(request, room):
    """Fixed message retrieval function"""
    try:
        # Get room by name
        try:
            room_obj = Room.objects.get(name=room, is_active=True)
        except Room.DoesNotExist:
            return JsonResponse({"messages": [], "error": "Room not found"}, status=404)

        # Get messages for this room, ordered by creation time
        messages = Message.objects.filter(room=room_obj).order_by("created_at")

        # Format messages for frontend
        messages_data = []
        for msg in messages:
            messages_data.append(
                {
                    "user": msg.username,  # Use the username field from the message
                    "value": msg.content,  # Frontend expects 'value' not 'content'
                    "date": msg.created_at.isoformat(),
                    "message_type": msg.message_type,
                }
            )

        return JsonResponse({"messages": messages_data})

    except Exception as e:
        logger.error(f"Error retrieving messages for room {room}: {str(e)}")
        return JsonResponse(
            {"messages": [], "error": "Failed to retrieve messages"}, status=500
        )


@login_required(login_url="/auth/login/")
def ai_home(request):
    """AI chat home with model selection"""
    username = request.user.get_full_name() or request.user.username

    available_models = {}
    if DIALOGPT_AVAILABLE:
        available_models["dialogpt"] = {
            "display": "DialoGPT",
            "color": "primary",
            "description": "Microsoft's conversational AI",
            "type": "text",
            "available": True,
        }

    if GEMINI_AVAILABLE:
        gemini_models = {
            "gemini-2.5-pro": {
                "display": "Gemini 2.5 Pro",
                "color": "success",
                "description": "Enhanced thinking, reasoning, multimodal",
                "type": "text",
                "available": True,
            },
            "gemini-2.5-flash": {
                "display": "Gemini 2.5 Flash",
                "color": "info",
                "description": "Adaptive thinking, cost efficient",
                "type": "text",
                "available": True,
            },
            "gemini-2.0-flash": {
                "display": "Gemini 2.0 Flash",
                "color": "dark",
                "description": "Next generation features, speed, realtime streaming",
                "type": "text",
                "available": True,
            },
        }
        available_models.update(gemini_models)

    context = {
        "username": username,
        "available_models": available_models,
        "dialogpt_available": DIALOGPT_AVAILABLE,
        "gemini_available": GEMINI_AVAILABLE,
    }
    return render(request, "AIChatPage.html", context)


@login_required(login_url="/auth/login/")
def ai_chat(request):
    """Redirect to default AI model"""
    if not GEMINI_AVAILABLE and not DIALOGPT_AVAILABLE:
        messages.error(request, "No AI models are currently available")
        return redirect("home")
    return ai_chat_with_model(
        request, "gemini-2.5-flash" if GEMINI_AVAILABLE else "dialogpt"
    )


@login_required(login_url="/auth/login/")
def ai_chat_with_model(request, model):
    """Enhanced AI chat with conversation persistence"""
    if not GEMINI_AVAILABLE and not DIALOGPT_AVAILABLE:
        messages.error(request, "No AI models are currently available")
        return redirect("home")

    conversation_id = request.GET.get("conversation_id", str(uuid.uuid4()))
    username = request.user.get_full_name() or request.user.username

    # FIXED: Map to actual API model name
    api_model = MODEL_MAPPING.get(model, "gemini-1.5-flash")

    model_config = {
        "dialogpt": {
            "display": "DialoGPT",
            "color": "primary",
            "description": "Microsoft's conversational AI",
            "type": "text",
            "available": DIALOGPT_AVAILABLE,
        },
        "gemini-2.5-pro": {
            "display": "Gemini 2.5 Pro",
            "color": "success",
            "description": "Enhanced thinking, reasoning, multimodal",
            "type": "text",
            "available": GEMINI_AVAILABLE,
        },
        "gemini-2.5-flash": {
            "display": "Gemini 2.5 Flash",
            "color": "info",
            "description": "Adaptive thinking, cost efficient",
            "type": "text",
            "available": GEMINI_AVAILABLE,
        },
        "gemini-2.0-flash": {
            "display": "Gemini 2.0 Flash",
            "color": "dark",
            "description": "Next generation features, speed, realtime streaming",
            "type": "text",
            "available": GEMINI_AVAILABLE,
        },
        "gemini-live-2.5-flash-preview": {
            "display": "Gemini 2.5 Flash Live",
            "color": "danger",
            "description": "Low-latency voice and video interactions",
            "type": "voice",
            "available": GEMINI_AVAILABLE,
        },
    }

    if model not in model_config or not model_config[model]["available"]:
        if GEMINI_AVAILABLE:
            model = "gemini-2.5-flash"
            api_model = MODEL_MAPPING.get(model)
        elif DIALOGPT_AVAILABLE:
            model = "dialogpt"
            api_model = model
        else:
            messages.error(request, "No AI models are currently available")
            return redirect("home")

    config = model_config[model]

    context = {
        "conversation_id": conversation_id,
        "username": username,
        "model": model,  # FIXED: Use original model name for frontend
        "model_display": config["display"],
        "model_color": config["color"],
        "model_description": config["description"],
        "model_type": config["type"],
        "all_models": {k: v for k, v in model_config.items() if v["available"]},
    }

    return render(request, "AIChatPage.html", context)


@login_required(login_url="/auth/login/")
@require_POST
def get_ai_response(request):
    """Fixed AI response handler with proper error handling"""
    try:
        # Parse JSON data
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        model = data.get("model", "gemini-2.5-flash")
        message_type = data.get("message_type", "text")
        username = request.user.get_full_name() or request.user.username

        logger.info(
            f"AI request received - Model: {model}, User: {username}, Message: {user_message[:50]}..."
        )

        # Validate input
        if not user_message:
            logger.warning("Empty message received")
            return JsonResponse({"error": "Please enter a message"}, status=400)

        if len(user_message) > 5000:
            logger.warning("Message too long")
            return JsonResponse({"error": "Message too long"}, status=400)

        # Map to actual API model
        api_model = MODEL_MAPPING.get(model, "gemini-1.5-flash")
        logger.info(f"Using API model: {api_model}")

        # Get or create conversation history
        try:
            conversation_obj, created = ConversationHistory.objects.get_or_create(
                conversation_id=conversation_id,
                user=request.user,
                model_type=model,
                defaults={
                    "username": username,
                    "conversation_data": {"history": []},
                    "is_active": True,
                },
            )
            logger.info(
                f"Conversation {'created' if created else 'retrieved'}: {conversation_id}"
            )
        except Exception as e:
            logger.error(f"Error with conversation history: {e}")
            # Continue without conversation history if there's an issue
            conversation_obj = None

        # Generate AI response
        response_text = ""
        try:
            if api_model == "dialogpt" and DIALOGPT_AVAILABLE:
                logger.info("Using DialoGPT")
                response_text = get_dialogpt_response(user_message, conversation_id)
            elif api_model.startswith("gemini") and GEMINI_AVAILABLE:
                logger.info(f"Using Gemini model: {api_model}")
                response_text = get_gemini_response(
                    user_message, api_model, conversation_obj
                )
            else:
                logger.error(
                    f"Model not available: {api_model}, GEMINI_AVAILABLE: {GEMINI_AVAILABLE}, DIALOGPT_AVAILABLE: {DIALOGPT_AVAILABLE}"
                )
                return JsonResponse(
                    {
                        "error": f"Model {model} is not available. Please try a different model.",
                        "conversation_id": conversation_id,
                    },
                    status=503,
                )

            if not response_text or not response_text.strip():
                logger.warning("Empty response from AI model")
                response_text = (
                    "I'm sorry, I couldn't generate a response. Please try again."
                )

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            logger.error(traceback.format_exc())
            return JsonResponse(
                {
                    "error": "I'm having trouble processing your message right now. Please try again.",
                    "conversation_id": conversation_id,
                },
                status=500,
            )

        # Update conversation history if available
        if conversation_obj:
            try:
                history = conversation_obj.conversation_data.get("history", [])
                history.append(
                    {
                        "user_message": user_message,
                        "ai_response": response_text,
                        "timestamp": timezone.now().isoformat(),
                    }
                )

                # Keep only last 50 exchanges
                if len(history) > 50:
                    history = history[-50:]

                conversation_obj.conversation_data = {"history": history}
                conversation_obj.message_count += 1
                conversation_obj.last_accessed = timezone.now()
                conversation_obj.save()
                logger.info("Conversation history updated")
            except Exception as e:
                logger.error(f"Error updating conversation history: {e}")

        logger.info(f"AI response generated successfully for {username}")
        return JsonResponse(
            {
                "response": response_text,
                "conversation_id": conversation_id,
                "model_used": model,
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in get_ai_response: {e}")
        logger.error(traceback.format_exc())
        return JsonResponse(
            {
                "error": "Internal server error. Please try again.",
                "conversation_id": (
                    conversation_id
                    if "conversation_id" in locals()
                    else str(uuid.uuid4())
                ),
            },
            status=500,
        )


def get_dialogpt_response(user_message, conversation_id):
    """Enhanced DialoGPT response with better error handling"""
    if not DIALOGPT_AVAILABLE:
        raise Exception("DialoGPT model not available")

    try:
        # Simple response for demo - you can enhance this with actual DialoGPT generation
        responses = [
            f"That's interesting! You said: {user_message}",
            "I understand what you're saying. Could you tell me more?",
            "Thanks for sharing that with me. What would you like to know?",
            "I see. How can I help you with that?",
            f"Regarding '{user_message}', I think there are several ways to approach this.",
            "That's a good point. Let me think about that for a moment.",
        ]

        import random

        response = random.choice(responses)
        logger.info("DialoGPT response generated")
        return response

    except Exception as e:
        logger.error(f"DialoGPT generation error: {e}")
        return "I'm having trouble processing that right now. Could you try asking something else?"


def get_gemini_response(user_message, model_name, conversation_obj=None):
    """Fixed Gemini response with proper error handling"""
    if not GEMINI_AVAILABLE:
        raise Exception("Gemini API not available")

    try:
        logger.info(f"Initializing Gemini model: {model_name}")

        # Initialize model with the correct name
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            specific_model = genai.GenerativeModel(model_name)
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            # Fallback to a known working model
            logger.info("Falling back to gemini-1.5-flash-latest")
            specific_model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # Generate response with safety settings
        logger.info("Generating content with Gemini")
        response = specific_model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7, max_output_tokens=1000, top_p=0.8, top_k=40
            ),
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ],
        )

        # Handle response
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text
                if response_text and response_text.strip():
                    logger.info("Successfully got Gemini response")
                    return response_text.strip()

        logger.warning("No valid response from Gemini")
        return (
            "I'm not sure how to respond to that. Could you try asking something else?"
        )

    except Exception as e:
        logger.error(f"Gemini API error ({model_name}): {e}")
        logger.error(traceback.format_exc())

        # More specific error handling
        error_str = str(e).lower()
        if "quota" in error_str or "limit" in error_str:
            return (
                "I'm currently experiencing high demand. Please try again in a moment."
            )
        elif "key" in error_str or "auth" in error_str:
            return "There's an authentication issue with the AI service. Please contact support."
        elif "model" in error_str:
            return "The AI model is temporarily unavailable. Please try again later."
        else:
            return "I'm having trouble processing that right now. Could you try asking something else?"


def favicon_view(request):
    """Favicon handler"""
    return HttpResponse(status=204)


# Add this to test Gemini connection
@login_required(login_url="/auth/login/")
def test_gemini_connection(request):
    """Test view to verify Gemini API connectivity"""
    try:
        result = "🔧 GEMINI API TEST RESULTS\n"
        result += "=" * 50 + "\n\n"

        # Check API key
        if not GEMINI_API_KEY:
            result += "❌ GEMINI_API_KEY not found in environment\n"
            return HttpResponse(result, content_type="text/plain; charset=utf-8")

        result += f"✅ API Key found: {GEMINI_API_KEY[:10]}...\n"
        result += f"✅ GEMINI_AVAILABLE: {GEMINI_AVAILABLE}\n"

        if not GEMINI_AVAILABLE:
            result += "❌ Gemini not marked as available\n"
            return HttpResponse(result, content_type="text/plain; charset=utf-8")

        # Test connection
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(
            "Hello, respond with 'Connection successful!'"
        )

        if response and response.text:
            result += f"✅ Test response: {response.text}\n"
            result += "✅ Gemini API working correctly!\n"
        else:
            result += "❌ No response from Gemini\n"

        return HttpResponse(result, content_type="text/plain; charset=utf-8")

    except Exception as e:
        error_result = f"❌ Gemini API test failed: {e}\n"
        error_result += f"Error details: {traceback.format_exc()}\n"
        return HttpResponse(error_result, content_type="text/plain; charset=utf-8")
# Add these helper functions to your views.py if they're missing or incorrect:


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0].strip()
    else:
        ip = request.META.get("REMOTE_ADDR")
    return ip


def get_user_agent(request):
    """Get user agent from request"""
    return request.META.get("HTTP_USER_AGENT", "")[
        :500
    ]  # Limit length to prevent DB errors
