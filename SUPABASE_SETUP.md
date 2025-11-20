# Supabase Configuration Guide

This document explains how to configure Supabase for the Care-AI application.

## ⚠️ IMPORTANT: Database Setup Required

**Before using the Medication Manager**, you MUST create the required database tables in Supabase. The app will fail with "medications table does not exist" error if you skip this step.

### Quick Start - Create the Medications Table

1. Go to your Supabase Dashboard → SQL Editor
2. Run this SQL command:

```sql
-- Create medications table
CREATE TABLE medications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL DEFAULT auth.uid(),
  medication_data JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE medications ENABLE ROW LEVEL SECURITY;

-- Create policy for users to access only their own medications
CREATE POLICY "Users can manage their own medications"
  ON medications
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);
```

3. Click "Run" to create the table
4. You can now use the Medication Manager feature!

---

## Setup Instructions

### 1. Local Development

Create a `.streamlit/secrets.toml` file in the root directory with your Supabase credentials:

```toml
# Supabase Configuration
SUPABASE_URL = "https://okxbdihgnxwwfrgopbbw.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
```

**Note:** The `.streamlit/secrets.toml` file is already in `.gitignore` to prevent accidentally committing secrets.

### 2. Production/Cloud Deployment

For Streamlit Cloud or other hosting platforms:

1. Go to your app settings
2. Navigate to "Secrets" section
3. Add the following secrets:

```toml
SUPABASE_URL = "https://okxbdihgnxwwfrgopbbw.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
```

### 3. Required Database Tables

The application requires the following Supabase tables:

#### `medications` table
```sql
CREATE TABLE medications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL DEFAULT auth.uid(),
  medication_data JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE medications ENABLE ROW LEVEL SECURITY;

-- Create policy for users to access only their own medications
CREATE POLICY "Users can manage their own medications"
  ON medications
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);
```

#### Other Required Tables
- `symptom_history` - For symptom analysis storage
- `chat_conversations` - For chatbot conversations
- `chat_messages` - For chat message storage

Refer to the existing database schema for complete table definitions.

## Current Configuration

- **Supabase Project URL:** https://okxbdihgnxwwfrgopbbw.supabase.co
- **Anon/Public Key:** Configured in secrets.toml (not committed to repo)

## Troubleshooting

### Connection Issues
If you encounter connection errors:
1. Verify your SUPABASE_URL is correct
2. Check that your SUPABASE_KEY is the anon/public key (not the service role key)
3. Ensure the secrets.toml file exists in the `.streamlit/` directory

### Authentication Issues
If users cannot log in:
1. Check that email authentication is enabled in Supabase dashboard
2. Verify Row Level Security policies are properly configured
3. Ensure the auth.users table is accessible

### Email Verification Link Issues
If clicking verification links shows "This site can't be reached":

1. **Configure Site URL in Supabase Dashboard:**
   - Go to Authentication → URL Configuration in Supabase dashboard
   - Set **Site URL** to your app's URL:
     - For local development: `http://localhost:8501`
     - For Streamlit Cloud: `https://your-app-name.streamlit.app`
   
2. **Add Redirect URLs:**
   - In the same URL Configuration section
   - Add your app URL to **Redirect URLs** list:
     - Local: `http://localhost:8501`
     - Production: `https://your-app-name.streamlit.app`
   
3. **Email Template Configuration:**
   - Go to Authentication → Email Templates
   - Ensure the confirmation email uses `{{ .ConfirmationURL }}` correctly
   - The default template should work, but verify the link points to your Site URL

4. **Disable Email Confirmation (Optional - for development only):**
   - Go to Authentication → Providers → Email
   - Toggle off "Confirm email" if you want to skip verification during development
   - **Warning:** Re-enable this for production!

5. **Test the Fix:**
   - Clear browser cache/cookies
   - Sign up with a new email
   - Check that the verification link now works

## Security Notes

- **Never commit the `secrets.toml` file** - it's already in `.gitignore`
- The anon/public key is safe to use client-side (it has limited permissions)
- For sensitive operations, use Row Level Security (RLS) policies
- All user data is automatically scoped to the authenticated user via RLS
