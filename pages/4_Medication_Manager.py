# pages/4_Medication_Manager.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime, timedelta
from medication_manager import MedicationManager
from supabase_client import (
    get_supabase_client, 
    save_medication, 
    get_medications,
    update_medication,
    delete_medication
)
from style_utils import add_custom_css

# --- Page Configuration & Styling ---
st.set_page_config(page_title="Medication Manager", page_icon="💊")
add_custom_css()
st.title("💊 Medication Manager")

# --- Authentication & Supabase Client ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access the Medication Manager.")
    st.stop()
    
supabase = get_supabase_client()

# --- Create Tabs ---
tab1, tab2 = st.tabs(["My Medications", "Add Medication"])

# --- Tab 1: View and Manage Medications ---
with tab1:
    st.header("Your Medications")
    st.info("Track your medications and get reminders when it's time to take them.", icon="ℹ️")
    
    # Fetch medications from database
    medications = get_medications(supabase)
    
    if not medications:
        st.info("You haven't added any medications yet. Use the 'Add Medication' tab to get started.", icon="👉")
    else:
        # Display each medication in an expander
        for med in medications:
            # Calculate dose status
            is_reminder_due, is_dose_due, time_until = MedicationManager.is_dose_due(
                med['schedule_time'], 
                med['reminder_minutes']
            )
            
            # Create status indicator
            if is_dose_due:
                status_color = "🔴"
                status_text = "**DOSE DUE NOW!**"
            elif is_reminder_due:
                status_color = "🟡"
                status_text = f"**Reminder:** Take in {MedicationManager.format_time_until(time_until)}"
            else:
                status_color = "🟢"
                status_text = f"Next dose in {MedicationManager.format_time_until(time_until)}"
            
            # Display medication card
            with st.expander(f"{status_color} **{med['name']}** - {med['dosage']}", expanded=is_dose_due or is_reminder_due):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Dosage:** {med['dosage']}")
                    st.markdown(f"**Schedule:** {med['schedule_time']}")
                    st.markdown(f"**Reminder:** {MedicationManager.get_reminder_label(med['reminder_minutes'])} before")
                    st.markdown(f"**Status:** {status_text}")
                    
                    if is_dose_due:
                        st.error("⚠️ It's time to take this medication!", icon="⏰")
                    elif is_reminder_due:
                        st.warning(f"⏰ Don't forget to take your medication soon!", icon="🔔")
                
                with col2:
                    # Edit button
                    if st.button("✏️ Edit", key=f"edit_{med['id']}"):
                        st.session_state.editing_med_id = med['id']
                        st.session_state.edit_name = med['name']
                        st.session_state.edit_dosage = med['dosage']
                        st.session_state.edit_schedule = med['schedule_time']
                        st.session_state.edit_reminder = med['reminder_minutes']
                        st.rerun()
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{med['id']}"):
                        if delete_medication(supabase, med['id']):
                            st.success("Medication deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete medication.")
                
                # Show edit form if this medication is being edited
                if 'editing_med_id' in st.session_state and st.session_state.editing_med_id == med['id']:
                    st.markdown("---")
                    st.subheader("Edit Medication")
                    
                    with st.form(f"edit_form_{med['id']}"):
                        edit_name = st.text_input("Medication Name", value=st.session_state.edit_name)
                        edit_dosage = st.text_input("Dosage", value=st.session_state.edit_dosage)
                        edit_schedule = st.time_input("Schedule Time", value=datetime.strptime(st.session_state.edit_schedule, "%H:%M").time())
                        
                        # Reminder selection
                        reminder_labels = list(MedicationManager.REMINDER_OPTIONS.keys())
                        current_reminder_label = MedicationManager.get_reminder_label(st.session_state.edit_reminder)
                        current_index = reminder_labels.index(current_reminder_label) if current_reminder_label in reminder_labels else 0
                        
                        edit_reminder_choice = st.selectbox(
                            "Reminder Time",
                            options=reminder_labels,
                            index=current_index
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Save Changes"):
                                # Validate inputs
                                if not edit_name or not edit_dosage:
                                    st.error("Please fill in all fields.")
                                else:
                                    # Update medication
                                    updated_med = MedicationManager.create_medication(
                                        edit_name,
                                        edit_dosage,
                                        edit_schedule.strftime("%H:%M"),
                                        MedicationManager.REMINDER_OPTIONS[edit_reminder_choice]
                                    )
                                    
                                    if update_medication(supabase, med['id'], updated_med):
                                        st.success("Medication updated successfully!")
                                        # Clear editing state
                                        del st.session_state.editing_med_id
                                        st.rerun()
                                    else:
                                        st.error("Failed to update medication.")
                        
                        with col2:
                            if st.form_submit_button("Cancel"):
                                del st.session_state.editing_med_id
                                st.rerun()

# --- Tab 2: Add New Medication ---
with tab2:
    st.header("Add New Medication")
    st.write("Enter the details of your medication below.")
    
    with st.form("add_medication_form", clear_on_submit=True):
        med_name = st.text_input(
            "Medication Name*",
            placeholder="e.g., Aspirin, Lisinopril",
            help="Enter the name of your medication"
        )
        
        med_dosage = st.text_input(
            "Dosage*",
            placeholder="e.g., 10mg, 2 tablets, 5ml",
            help="Specify the amount to take"
        )
        
        med_schedule = st.time_input(
            "Schedule Time*",
            help="What time should you take this medication?"
        )
        
        reminder_choice = st.selectbox(
            "Reminder Time*",
            options=list(MedicationManager.REMINDER_OPTIONS.keys()),
            help="When should we remind you before your dose?"
        )
        
        st.markdown("---")
        submit_button = st.form_submit_button("💾 Save Medication")
        
        if submit_button:
            # Validate inputs
            if not med_name or not med_dosage:
                st.error("Please fill in all required fields (marked with *).")
            else:
                # Create medication record
                medication = MedicationManager.create_medication(
                    name=med_name,
                    dosage=med_dosage,
                    schedule_time=med_schedule.strftime("%H:%M"),
                    reminder_minutes=MedicationManager.REMINDER_OPTIONS[reminder_choice]
                )
                
                # Save to database
                success, result = save_medication(supabase, medication)
                
                if success:
                    st.success(f"✅ {med_name} has been added to your medication list!")
                    st.info(f"You will receive a reminder {reminder_choice} before your scheduled dose at {med_schedule.strftime('%I:%M %p')}.")
                    st.info("💡 Switch to the 'My Medications' tab to view all your medications.", icon="👉")
                else:
                    # Handle different error types
                    if result == "medications_table_missing":
                        st.error("⚠️ Database Error: The 'medications' table does not exist.")
                        st.info("**Setup Required:** Please create the medications table in Supabase. See SUPABASE_SETUP.md for SQL commands.", icon="📚")
                        st.markdown("**Quick Fix - Run this SQL command:**")
                        st.code("""
CREATE TABLE medications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL DEFAULT auth.uid(),
  medication_data JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE medications ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage their own medications"
  ON medications
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);
                        """, language="sql")
                    elif result == "rls_policy_error":
                        st.error("⚠️ Permission Error: Cannot save medication due to Row Level Security policy.")
                        st.info("Please check that RLS policies are correctly configured in Supabase. See SUPABASE_SETUP.md for details.", icon="🔒")
                    else:
                        st.error(f"Failed to save medication: {result}")
                        st.info("Check the SUPABASE_SETUP.md file for database setup instructions.", icon="📚")

# --- Auto-refresh for live reminders ---
st.markdown("---")
st.caption("💡 Tip: This page updates automatically to show current medication reminders. Keep it open to stay on track!")
