from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinLengthValidator, MaxLengthValidator
import uuid


class Room(models.Model):
    """Chat room model with proper relationships and metadata"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(
        max_length=100,
        unique=True,
        validators=[MinLengthValidator(3), MaxLengthValidator(100)],
        help_text="Room name must be between 3 and 100 characters",
    )
    description = models.TextField(max_length=500, blank=True, null=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_rooms",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    max_participants = models.PositiveIntegerField(default=50)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self):
        return self.name

    @property
    def participant_count(self):
        """Get the number of users who have sent messages in this room"""
        return Message.objects.filter(room=self).values("user").distinct().count()

    @property
    def last_activity(self):
        """Get the timestamp of the last message in this room"""
        last_message = Message.objects.filter(room=self).order_by("-created_at").first()
        return last_message.created_at if last_message else self.created_at


class Message(models.Model):
    """Enhanced message model with better relationships and metadata"""

    MESSAGE_TYPES = [
        ("text", "Text Message"),
        ("voice", "Voice Message"),
        ("ai_response", "AI Response"),
        ("system", "System Message"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content = models.TextField(
        max_length=10000,
        validators=[MinLengthValidator(1)],
        help_text="Message content",
        default="",
        blank=True,
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="messages", null=True, blank=True
    )
    username = models.CharField(
        max_length=150,
        help_text="Username for display (for anonymous users)",
        default="Anonymous",
        blank=True,
    )
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name="messages")
    message_type = models.CharField(
        max_length=20, choices=MESSAGE_TYPES, default="text"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_edited = models.BooleanField(default=False)
    edited_at = models.DateTimeField(null=True, blank=True)
    reply_to = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="replies"
    )

    # Metadata fields
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["room", "-created_at"]),
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["message_type"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.username} in {self.room.name}: {self.content[:50]}..."

    # Replace the save method in your Message model with this corrected version:


    def save(self, *args, **kwargs):
        # Only check for content changes if this is an update (not a new message)
        if self.pk:  # This means the object already exists in database
            try:
                original_message = Message.objects.get(pk=self.pk)
                if original_message.content != self.content:
                    self.is_edited = True
                    self.edited_at = timezone.now()
            except Message.DoesNotExist:
                # If for some reason the original doesn't exist, skip edit tracking
                pass

        # Call the parent save method
        super().save(*args, **kwargs)


class ConversationHistory(models.Model):
    """Store AI conversation histories with automatic cleanup"""

    CONVERSATION_TYPES = [
        ("dialogpt", "DialoGPT"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation_id = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="ai_conversations",
    )
    username = models.CharField(max_length=150, default="Anonymous", blank=True)
    model_type = models.CharField(max_length=50, choices=CONVERSATION_TYPES)
    conversation_data = models.JSONField(default=dict)  # Store conversation history
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    message_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-last_accessed"]
        indexes = [
            models.Index(fields=["conversation_id"]),
            models.Index(fields=["user", "-last_accessed"]),
            models.Index(fields=["model_type"]),
            models.Index(fields=["is_active", "-last_accessed"]),
        ]

    def __str__(self):
        return f"{self.username} - {self.model_type} ({self.message_count} messages)"


class UserProfile(models.Model):
    """Extended user profile for additional features"""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    avatar = models.ImageField(upload_to="avatars/", null=True, blank=True)
    bio = models.TextField(max_length=500, blank=True)
    preferred_ai_model = models.CharField(
        max_length=50,
        choices=ConversationHistory.CONVERSATION_TYPES,
        default="gemini-2.5-flash",
    )
    voice_enabled = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Usage statistics
    total_messages_sent = models.PositiveIntegerField(default=0)
    total_ai_conversations = models.PositiveIntegerField(default=0)
    last_active = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["last_active"]),
            models.Index(fields=["preferred_ai_model"]),
        ]

    def __str__(self):
        return f"{self.user.username}'s Profile"


class RoomParticipant(models.Model):
    """Track room participation"""

    room = models.ForeignKey(
        Room, on_delete=models.CASCADE, related_name="participants"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="room_memberships"
    )
    joined_at = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    role = models.CharField(
        max_length=20,
        choices=[
            ("member", "Member"),
            ("moderator", "Moderator"),
            ("admin", "Admin"),
        ],
        default="member",
    )

    class Meta:
        unique_together = ("room", "user")
        indexes = [
            models.Index(fields=["room", "is_active"]),
            models.Index(fields=["user", "last_seen"]),
        ]

    def __str__(self):
        return f"{self.user.username} in {self.room.name}"


class SystemSettings(models.Model):
    """Store system-wide settings"""

    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # fixed line
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Setting"
        verbose_name_plural = "System Settings"

    def __str__(self):
        return f"{self.key}: {self.value[:50]}"
