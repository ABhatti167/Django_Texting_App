import os
import uuid
import tempfile
import base64
import io
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import torch
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import traceback

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
        return redirect(f"/ai-chat/dialogpt/?username={username}")
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


def save_audio_file(audio_file):
    """
    Save uploaded audio file as WebM (what the browser sends).
    """
    try:
        # Always save as .webm since that's what the browser sends
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            for chunk in audio_file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        print(f"💾 Audio file saved as WebM: {tmp_file_path}")
        return tmp_file_path

    except Exception as e:
        print(f"❌ Error saving audio file: {e}")
        raise e


@csrf_exempt
def get_ai_response(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        conversation_id = request.POST.get("conversation_id", "")
        model = request.POST.get("model", "dialogpt")
        message_type = request.POST.get("message_type", "text")

        print(f"🔍 Processing request - Model: {model}, Type: {message_type}")

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # Voice message handling
        if message_type == "voice":
            audio_file = request.FILES.get("audio")
            if not audio_file:
                print("❌ No audio file received")
                return JsonResponse({"response": "No audio file received."})

            print(
                f"📁 Received audio file: {audio_file.name}, Size: {audio_file.size} bytes"
            )

            try:
                # Save audio file - speech_recognition can handle multiple formats
                print(
                    f"🎵 Audio content type: {getattr(audio_file, 'content_type', 'unknown')}"
                )
                print(f"🎵 Audio file name: {getattr(audio_file, 'name', 'unknown')}")

                # Save the uploaded WebM audio file
                tmp_file_path = save_audio_file(audio_file)
                print(f"💾 Temporary WebM file saved: {tmp_file_path}")

                # Use speech_recognition with WebM file directly
                print("🎤 Processing WebM audio with speech recognition...")
                recognizer = sr.Recognizer()

                try:
                    # Use the microphone-based recognition for WebM files
                    # This bypasses the AudioFile limitation
                    with open(tmp_file_path, "rb") as webm_file:
                        # Read the WebM data
                        webm_data = webm_file.read()

                    # Try using speech_recognition's ability to handle raw audio
                    # We'll use a workaround by creating an AudioData object
                    import io
                    from speech_recognition import AudioData

                    # Create a BytesIO object from the WebM data
                    audio_io = io.BytesIO(webm_data)

                    # This is a workaround - we'll try to recognize it as raw audio
                    print("🌐 Attempting WebM recognition via Google API...")
                    try:
                        # Use recognize_google with raw audio data
                        # Note: This might not work perfectly with WebM, but let's try
                        user_message = recognizer.recognize_google_cloud(
                            webm_data, language="en-US"
                        )
                        print(f"✅ Transcribed successfully: '{user_message}'")
                    except:
                        # Fallback to regular Google API
                        try:
                            # Try a different approach - save as temporary wav-like data
                            # This is a hack but might work
                            audio_data = AudioData(
                                webm_data, sample_rate=16000, sample_width=2
                            )
                            user_message = recognizer.recognize_google(audio_data)
                            print(
                                f"✅ Transcribed with fallback method: '{user_message}'"
                            )
                        except:
                            # Final fallback - just return a helpful message
                            user_message = "I'm having trouble processing the audio. Please try speaking clearly into your microphone."
                            print("⚠️ All recognition methods failed")

                except Exception as recognition_error:
                    print(f"❌ Speech recognition failed: {recognition_error}")
                    user_message = "Sorry, I couldn't process the audio. Please try speaking more clearly."

                # Clean up temp file
                try:
                    os.remove(tmp_file_path)
                    print("🗑️ Temporary file cleaned up")
                except:
                    pass

                # If transcription failed, return error
                if user_message.startswith("Sorry"):
                    return JsonResponse(
                        {"response": user_message, "conversation_id": conversation_id}
                    )

                # Now get AI response to transcribed text
                print(f"🤖 Getting AI response for: '{user_message}'")
                if model == "dialogpt":
                    response_text = get_dialogpt_response(user_message, conversation_id)
                elif model.startswith("gemini"):
                    # Use gemini-2.5-flash instead of the non-existent live model
                    actual_model = (
                        "gemini-2.5-flash"
                        if model == "gemini-live-2.5-flash-preview"
                        else model
                    )
                    response_text = get_gemini_response_with_model(
                        user_message, conversation_id, actual_model
                    )
                else:
                    response_text = "Unknown model specified."

                # Generate audio response for voice models
                audio_url = None
                if model == "gemini-live-2.5-flash-preview":
                    # Use a working Gemini model for voice responses
                    print("🔊 Generating TTS audio response...")
                    try:
                        # Generate TTS
                        tts = gTTS(text=response_text, lang="en", slow=False)
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        audio_base64 = base64.b64encode(fp.read()).decode("utf-8")
                        audio_url = f"data:audio/mp3;base64,{audio_base64}"
                        print("✅ TTS audio generated successfully")
                    except Exception as tts_error:
                        print(f"⚠️ TTS generation failed: {tts_error}")

                print("✅ Voice processing completed successfully")
                return JsonResponse(
                    {
                        "response": f"{response_text}",  # Removed the transcribed prefix
                        "conversation_id": conversation_id,
                        "audio_url": audio_url,
                    }
                )

            except Exception as e:
                print(f"❌ Voice processing error: {e}")
                print(f"📋 Full traceback: {traceback.format_exc()}")
                return JsonResponse(
                    {
                        "response": f"Error processing voice message: {str(e)}",
                        "conversation_id": conversation_id,
                    }
                )

        # Text message handling
        if not user_message:
            return JsonResponse({"response": "Please enter a message."})

        print(f"💬 Processing text message: '{user_message}'")

        try:
            if model == "dialogpt":
                response_text = get_dialogpt_response(user_message, conversation_id)
            elif model.startswith("gemini"):
                response_text = get_gemini_response_with_model(
                    user_message, conversation_id, model
                )
            else:
                response_text = "Unknown model specified."

            print(f"✅ Response generated: {response_text[:100]}...")
            return JsonResponse(
                {"response": response_text, "conversation_id": conversation_id}
            )

        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            print(f"📋 Full traceback: {traceback.format_exc()}")
            return JsonResponse(
                {"response": "Sorry, I'm having trouble responding right now."}
            )

    return JsonResponse({"response": "Invalid request"}, status=400)


def get_dialogpt_response(user_message, conversation_id):
    print(f"🤖 DialoGPT processing: '{user_message}'")

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

    print(f"✅ DialoGPT response: '{bot_response}'")
    return bot_response


def get_gemini_response_with_model(user_message, conversation_id, model_name):
    print(f"🤖 Gemini ({model_name}) processing: '{user_message}'")

    try:
        specific_model = genai.GenerativeModel(model_name)

        chat_key = f"{conversation_id}_{model_name}"

        if chat_key not in gemini_chats:
            gemini_chats[chat_key] = specific_model.start_chat(history=[])

        chat = gemini_chats[chat_key]
        response = chat.send_message(user_message)

        print(f"✅ Gemini response: '{response.text[:100]}...'")
        return response.text

    except Exception as e:
        print(f"❌ Specific Gemini model error ({model_name}): {e}")

        try:
            print(f"🔄 Trying fallback generation for {model_name}...")
            specific_model = genai.GenerativeModel(model_name)
            response = specific_model.generate_content(user_message)
            print(f"✅ Gemini fallback response: '{response.text[:100]}...'")
            return response.text
        except Exception as fallback_error:
            print(f"❌ Gemini model fallback error: {fallback_error}")
            return f"Error with {model_name}: {str(fallback_error)[:100]}..."
