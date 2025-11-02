# Custom GPT Framework

A flexible, organization-ready GPT tool built with Streamlit and OpenAI that can be easily customized for any organization's needs.

## Features

- **Role-Based Access Control**: Admin and public user roles with different permissions
- **Admin Features**: App configuration, document upload, and knowledge base management
- **Public User Features**: Chat functionality with existing knowledge base
- **Dynamic Branding**: Easily customize app name, icon, and organization branding
- **RAG (Retrieval-Augmented Generation)**: Upload documents or crawl websites to create a custom knowledge base
- **Multiple Input Methods**: Support for file uploads (PDF, TXT, MD) and URL crawling
- **Persistent Vector Store**: Knowledge base persists between sessions using ChromaDB
- **Real-time Customization**: Change app settings through the sidebar interface

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your organization**
   - Copy `config_template.env` to `.env`
   - Add your OpenAI API key
   - Customize the app name, icon, and organization name

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Customization

### Example Configurations

**Healthcare Organization:**
```env
APP_NAME=MedAssist GPT
APP_ICON=🏥
ORGANIZATION_NAME=HealthCorp
```

**Educational Institution:**
```env
APP_NAME=EduBot
APP_ICON=🎓
ORGANIZATION_NAME=University
```

**Legal Firm:**
```env
APP_NAME=LegalAI
APP_ICON=⚖️
ORGANIZATION_NAME=LawFirm
```

### Runtime Customization

You can also customize the app name and organization through the sidebar interface:

1. **Temporary Changes**: Changes made in the sidebar persist for the current session only
2. **Permanent Changes**: Click "💾 Save Permanently" to write changes to the `.env` file
3. **Reset**: Click "🔄 Reset to Defaults" to revert to original `.env` values
4. **Apply Permanently**: Restart the app after saving to see changes in the browser tab title

**Note**: The main title and knowledge base header update immediately, but the browser tab title requires an app restart to reflect saved changes.

## Usage

### First Time Setup
1. **Admin Setup**: Login with admin credentials to configure the app and upload initial documents
2. **Public Access**: Share the app with users who can chat without admin privileges

### Admin Functions
- **Configure Settings**: Customize app name, organization, and system prompts
- **Manage Knowledge Base**: Upload files or add URLs to build the organization's knowledge base
- **Save Configuration**: Make changes permanent by saving to .env file

### Public User Functions  
- **Chat Interface**: Ask questions and get answers based on the uploaded knowledge base
- **No Configuration**: Cannot modify app settings or upload documents

### Role-Based Access
- **Admin Login**: Use username/password to access full functionality
- **Public Mode**: Continue as public user for chat-only access
- **Easy Switching**: Switch between modes without restarting the app

## File Structure

```
AI/
├── app.py                 # Main Streamlit application
├── rag_utils.py          # RAG utilities for document processing
├── requirements.txt      # Python dependencies
├── config_template.env   # Configuration template
├── .env                  # Your configuration (create from template)
├── README.md            # This file
├── data/                # Directory for data files
├── tmp_uploads/         # Temporary file storage
└── .chroma/             # Vector database storage (created automatically)
```

## Contributing

Feel free to submit issues and enhancement requests!