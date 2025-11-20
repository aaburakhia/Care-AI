# medication_manager.py
"""
Medication Manager Module
Provides functionality for tracking medications, dosages, schedules, and reminders.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class MedicationManager:
    """Handles medication tracking and reminder scheduling."""
    
    REMINDER_OPTIONS = {
        "15 minutes": 15,
        "30 minutes": 30,
        "1 hour": 60
    }
    
    @staticmethod
    def create_medication(name: str, dosage: str, schedule_time: str, reminder_minutes: int) -> Dict:
        """
        Create a medication record.
        
        Args:
            name: Name of the medication
            dosage: Dosage information (e.g., "10mg", "2 tablets")
            schedule_time: Time when medication should be taken (HH:MM format)
            reminder_minutes: Minutes before dose time to send reminder
            
        Returns:
            Dictionary containing medication details
        """
        return {
            "name": name,
            "dosage": dosage,
            "schedule_time": schedule_time,
            "reminder_minutes": reminder_minutes,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
    
    @staticmethod
    def validate_time_format(time_str: str) -> bool:
        """
        Validate that time string is in HH:MM format.
        
        Args:
            time_str: Time string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            datetime.strptime(time_str, "%H:%M")
            return True
        except ValueError:
            return False
    
    @staticmethod
    def calculate_next_dose_time(schedule_time: str) -> datetime:
        """
        Calculate the next dose time based on schedule.
        
        Args:
            schedule_time: Time in HH:MM format
            
        Returns:
            DateTime object for next dose
        """
        now = datetime.now()
        hour, minute = map(int, schedule_time.split(":"))
        
        next_dose = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time has already passed today, schedule for tomorrow
        if next_dose <= now:
            next_dose += timedelta(days=1)
        
        return next_dose
    
    @staticmethod
    def calculate_reminder_time(dose_time: datetime, reminder_minutes: int) -> datetime:
        """
        Calculate when reminder should be sent.
        
        Args:
            dose_time: DateTime when dose is due
            reminder_minutes: Minutes before dose to send reminder
            
        Returns:
            DateTime for reminder
        """
        return dose_time - timedelta(minutes=reminder_minutes)
    
    @staticmethod
    def is_dose_due(schedule_time: str, reminder_minutes: int) -> tuple:
        """
        Check if a dose is currently due or if reminder should be shown.
        
        Args:
            schedule_time: Time in HH:MM format
            reminder_minutes: Minutes before dose to send reminder
            
        Returns:
            Tuple of (is_reminder_due, is_dose_due, time_until_dose)
        """
        next_dose = MedicationManager.calculate_next_dose_time(schedule_time)
        reminder_time = MedicationManager.calculate_reminder_time(next_dose, reminder_minutes)
        now = datetime.now()
        
        is_reminder_due = reminder_time <= now < next_dose
        is_dose_due = now >= next_dose
        
        if now < next_dose:
            time_until_dose = next_dose - now
        else:
            time_until_dose = timedelta(0)
        
        return (is_reminder_due, is_dose_due, time_until_dose)
    
    @staticmethod
    def format_time_until(time_delta: timedelta) -> str:
        """
        Format a timedelta into human-readable string.
        
        Args:
            time_delta: Time difference to format
            
        Returns:
            Formatted string (e.g., "2 hours 30 minutes")
        """
        total_seconds = int(time_delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        if hours > 0 and minutes > 0:
            return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
        elif hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        elif minutes > 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            return "less than a minute"
    
    @staticmethod
    def get_reminder_label(minutes: int) -> str:
        """
        Get display label for reminder time.
        
        Args:
            minutes: Minutes before dose
            
        Returns:
            Display label
        """
        for label, value in MedicationManager.REMINDER_OPTIONS.items():
            if value == minutes:
                return label
        return f"{minutes} minutes"
