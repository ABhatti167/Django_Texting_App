import os
import uuid
import tempfile
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import torch
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment

from textpage.models import Room, Message

# Load environment variables from .env
load_dotenv()

# Initialize DialoGPT model and tokenizer once
dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Setup Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
gemini_model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.5-pro")
        GEMINI_AVAILABLE = True
        print("✅ Gemini API configured successfully with gemini-2.5-pro")
    except Exception:
        try:
            gemini_model = genai.GenerativeModel("gemini-pro")
            GEMINI_AVAILABLE = True
            print("✅ Gemini API configured successfully with gemini-pro (fallback)")
        except Exception:
            try:
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                GEMINI_AVAILABLE = True
                print(
                    "✅ Gemini API configured successfully with gemini-1.5-flash (fallback)"
                )
            except Exception as e3:
                print(f"❌ Failed to configure Gemini with any model: {e3}")
                GEMINI_AVAILABLE = False
else:
    print("❌ GEMINI_API_KEY not found in environment variables")

# In-memory conversation histories for stateful chat
user_histories = {}
gemini_chats = {}


def home(request):
    return render(request, "home.html")


def room(request, room):
    username = request.GET.get("username")
    room_details = Room.objects.get(name=room)
    return render(
        request,
        "room.html",
        {"username": username, "room": room, "room_details": room_details},
    )


def checkview(request):
    room = request.POST.get("room_name")
    username = request.POST.get("username")
    action = request.POST.get("action")

    if action == "enter_room":
        if Room.objects.filter(name=room).exists():
            return redirect("/room/" + room + "/?username=" + username)
        else:
            new_room = Room.objects.create(name=room)
            new_room.save()
            return redirect("/room/" + room + "/?username=" + username)
    elif action == "chat_ai":
        return redirect(f"/ai-chat/diablogpt/?username={username}")
    else:
        return redirect("/")


def send(request):
    message = request.POST["message"]
    username = request.POST["username"]
    room_id = request.POST["room_id"]
    Message.objects.create(value=message, user=username, room=room_id)
    return HttpResponse("Message sent successfully")


def getMessages(request, room):
    room_details = Room.objects.get(name=room)
    messages = Message.objects.filter(room=room_details.id)
    return JsonResponse({"messages": list(messages.values())})


def ai_home(request):
    username = request.GET.get("username", "Guest")
    return render(request, "AIChatPage.html", {"username": username})


def ai_chat(request):
    return ai_chat_with_model(request, "dialogpt")


def ai_chat_with_model(request, model):
    conversation_id = str(uuid.uuid4())
    username = request.GET.get("username", "Guest")

    model_config = {
        "dialogpt": {
            "display": "DialoGPT",
            "color": "primary",
            "description": "Microsoft's conversational AI",
            "type": "text",
        },
        "gemini-2.5-pro": {
            "display": "Gemini 2.5 Pro",
            "color": "success",
            "description": "Enhanced thinking, reasoning, multimodal",
            "type": "text",
        },
        "gemini-2.5-flash": {
            "display": "Gemini 2.5 Flash",
            "color": "info",
            "description": "Adaptive thinking, cost efficient",
            "type": "text",
        },
        "gemini-2.5-flash-lite": {
            "display": "Gemini 2.5 Flash Lite",
            "color": "warning",
            "description": "Most cost-efficient, high throughput",
            "type": "text",
        },
        "gemini-live-2.5-flash-preview": {
            "display": "Gemini 2.5 Flash Live",
            "color": "danger",
            "description": "Low-latency voice and video interactions",
            "type": "voice",
        },
        "gemini-2.0-flash": {
            "display": "Gemini 2.0 Flash",
            "color": "dark",
            "description": "Next generation features, speed, realtime streaming",
            "type": "text",
        },
        "gemini-2.0-flash-lite": {
            "display": "Gemini 2.0 Flash Lite",
            "color": "secondary",
            "description": "Cost efficiency and low latency",
            "type": "text",
        },
        "gemini-1.5-flash": {
            "display": "Gemini 1.5 Flash",
            "color": "outline-primary",
            "description": "Fast and versatile performance",
            "type": "text",
        },
    }

    if model not in model_config:
        model = "dialogpt"

    config = model_config[model]

    return render(
        request,
        "AIChatPage.html",
        {
            "conversation_id": conversation_id,
            "username": username,
            "model": model,
            "model_display": config["display"],
            "model_color": config["color"],
            "model_description": config["description"],
            "model_type": config["type"],
            "all_models": model_config,
        },
    )


@csrf_exempt
def get_ai_response(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        conversation_id = request.POST.get("conversation_id", "")
        model = request.POST.get("model", "dialogpt")
        message_type = request.POST.get("message_type", "text")

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Voice message handling: convert audio to text using SpeechRecognition + pydub
        if message_type == "voice":
            audio_file = request.FILES.get("audio")
            if not audio_file:
                return JsonResponse({"response": "No audio file received."})

            try:
                # Save the uploaded audio file temporarily as webm
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".webm"
                ) as tmp_file:
                    for chunk in audio_file.chunks():
                        tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name

                # Convert webm to wav using pydub (requires ffmpeg installed)
                wav_path = tmp_file_path.replace(".webm", ".wav")
                audio = AudioSegment.from_file(tmp_file_path)
                audio.export(wav_path, format="wav")

                # Use SpeechRecognition to transcribe audio
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    user_message = recognizer.recognize_google(audio_data)

                # Clean up temp files
                os.remove(tmp_file_path)
                os.remove(wav_path)

                # Now get AI response to transcribed text
                if model == "dialogpt":
                    response = get_dialogpt_response(user_message, conversation_id)
                elif model.startswith("gemini"):
                    response = get_gemini_response_with_model(
                        user_message, conversation_id, model
                    )
                else:
                    response = "Unknown model specified."

                return JsonResponse(
                    {"response": response, "conversation_id": conversation_id}
                )

            except Exception as e:
                print(f"Voice processing error: {e}")
                return JsonResponse({"response": "Error processing voice message."})

        # Text message handling
        if not user_message:
            return JsonResponse({"response": "Please enter a message."})

        try:
            if model == "dialogpt":
                response = get_dialogpt_response(user_message, conversation_id)
            elif model.startswith("gemini"):
                response = get_gemini_response_with_model(
                    user_message, conversation_id, model
                )
            else:
                response = "Unknown model specified."

            return JsonResponse(
                {"response": response, "conversation_id": conversation_id}
            )

        except Exception as e:
            print(f"Error with {model}: {e}")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
            return JsonResponse(
                {"response": "Sorry, I'm having trouble responding right now."}
            )

    return JsonResponse({"response": "Invalid request"}, status=400)


def get_dialogpt_response(user_message, conversation_id):
    chat_history_ids = user_histories.get(conversation_id, None)

    new_input_ids = dialogpt_tokenizer.encode(
        user_message + dialogpt_tokenizer.eos_token, return_tensors="pt"
    )

    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    chat_history_ids = dialogpt_model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=dialogpt_tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    user_histories[conversation_id] = chat_history_ids

    bot_response = dialogpt_tokenizer.decode(
        chat_history_ids[:, input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    return bot_response


def get_gemini_response(user_message, conversation_id):
    if not GEMINI_AVAILABLE or not gemini_model:
        print("DEBUG - Gemini not available or model not loaded")
        return "Gemini is not available."

    try:
        if conversation_id not in gemini_chats:
            gemini_chats[conversation_id] = gemini_model.start_chat(history=[])

        chat = gemini_chats[conversation_id]
        response = chat.send_message(user_message)
        return response.text

    except Exception as e:
        print(f"Gemini chat error: {e}")

        try:
            response = gemini_model.generate_content(user_message)
            return response.text
        except Exception:
            return "Gemini API error."


def get_gemini_response_with_model(user_message, conversation_id, model_name):
    try:
        specific_model = genai.GenerativeModel(model_name)

        chat_key = f"{conversation_id}_{model_name}"

        if chat_key not in gemini_chats:
            gemini_chats[chat_key] = specific_model.start_chat(history=[])

        chat = gemini_chats[chat_key]
        response = chat.send_message(user_message)

        return response.text

    except Exception as e:
        print(f"Specific Gemini model error ({model_name}): {e}")

        try:
            specific_model = genai.GenerativeModel(model_name)
            response = specific_model.generate_content(user_message)
            return response.text
        except Exception as fallback_error:
            print(f"Gemini model fallback error: {fallback_error}")
            return f"Error with {model_name}: {str(fallback_error)[:100]}..."
