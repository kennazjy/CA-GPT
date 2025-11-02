import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import requests

# NEW
from rag_utils import (
    get_chroma, get_collection, ingest_files_streaming,
    retrieve, build_context_snippets, answer_with_context
)

def save_settings_to_env(app_name_val, org_name_val):
    """Save app configuration to .env file"""
    try:
        env_path = ".env"
        
        # Read existing .env file
        env_lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        
        # Update or add the configuration variables
        updated_lines = []
        found_vars = {'APP_NAME': False, 'ORGANIZATION_NAME': False}
        
        for line in env_lines:
            stripped = line.strip()
            if stripped.startswith('APP_NAME='):
                updated_lines.append(f'APP_NAME={app_name_val}\n')
                found_vars['APP_NAME'] = True
            elif stripped.startswith('ORGANIZATION_NAME='):
                updated_lines.append(f'ORGANIZATION_NAME={org_name_val}\n')
                found_vars['ORGANIZATION_NAME'] = True
            else:
                updated_lines.append(line)
        
        # Add missing variables
        if not found_vars['APP_NAME']:
            updated_lines.append(f'APP_NAME={app_name_val}\n')
        if not found_vars['ORGANIZATION_NAME']:
            updated_lines.append(f'ORGANIZATION_NAME={org_name_val}\n')
        
        # Write back to .env file
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        return True
    except Exception as e:
        st.error(f"Error saving to .env: {str(e)}")
        return False

def check_admin_credentials(username, password):
    """Check admin credentials from .env file"""
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    return username == admin_username and password == admin_password

def show_login_form():
    """Display login form for admin access"""
    st.markdown("### 🔐 Admin Login")
    st.markdown("Please login to access admin features (app configuration and document upload)")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if check_admin_credentials(username, password):
                st.session_state.is_admin = True
                st.session_state.admin_username = username
                st.success("✅ Admin login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    
    st.markdown("---")
    st.markdown("### 👥 Public Access")
    if st.button("Continue as Public User"):
        st.session_state.is_admin = False
        st.session_state.user_role = "public"
        st.rerun()

def show_user_info():
    """Display current user info and logout option"""
    if st.session_state.get('is_admin', False):
        st.success(f"👨‍💼 Logged in as: {st.session_state.get('admin_username', 'Admin')}")
        if st.button("🚪 Logout"):
            st.session_state.is_admin = False
            if 'admin_username' in st.session_state:
                del st.session_state.admin_username
            if 'user_role' in st.session_state:
                del st.session_state.user_role
            st.rerun()
    else:
        st.info("👥 Public User Mode")
        if st.button("🔐 Switch to Admin"):
            # Clear all authentication state to show login form
            if 'user_role' in st.session_state:
                del st.session_state.user_role
            if 'is_admin' in st.session_state:
                del st.session_state.is_admin
            st.rerun()

load_dotenv()

# App configuration - make it dynamic for different organizations
app_name = st.secrets.get("APP_NAME", os.getenv("APP_NAME", "ECHO GPT"))
app_icon = st.secrets.get("APP_ICON", os.getenv("APP_ICON", "💬"))
organization_name = st.secrets.get("ORGANIZATION_NAME", os.getenv("ORGANIZATION_NAME", "ECHO"))

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to .env or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title=app_name, page_icon=app_icon)

# Authentication and role management
if 'is_admin' not in st.session_state and 'user_role' not in st.session_state:
    show_login_form()
    st.stop()

# Hide sidebar for public users using CSS
if not st.session_state.get('is_admin', False) and not st.session_state.get('show_sidebar_public', False):
    st.markdown("""
    <style>
    .css-1d391kg {display: none}
    .css-1rs6os {display: none}
    .css-17eq0hr {display: none}
    section[data-testid="stSidebar"] {display: none}
    section[data-testid="stSidebar"] > div {display: none}
    </style>
    """, unsafe_allow_html=True)

# Use runtime customization if available
display_app_name = st.session_state.get('custom_app_name', app_name)
display_organization = st.session_state.get('custom_organization', organization_name)

st.title(f"{app_icon} {display_app_name}")

# Show user info
show_user_info()

# Sidebar
with st.sidebar:
    # Show hide settings button for public users
    if not st.session_state.get('is_admin', False) and st.session_state.get('show_sidebar_public', False):
        if st.button("✖️ Hide Settings"):
            st.session_state.show_sidebar_public = False
            st.rerun()
        st.markdown("---")
    
    # Only show app configuration for admins
    if st.session_state.get('is_admin', False):
        st.markdown("### App Configuration")
        # Allow runtime customization - use session state values if they exist
        current_app_name = st.session_state.get('custom_app_name', app_name)
        current_organization = st.session_state.get('custom_organization', organization_name)
        
        runtime_app_name = st.text_input("App Name", value=current_app_name, help="Customize the app title")
        runtime_organization = st.text_input("Organization Name", value=current_organization, help="Customize organization name")
        
        st.caption("💡 Changes are temporary until saved permanently")
        
        # Update session state if changed and trigger rerun
        if runtime_app_name != current_app_name:
            st.session_state.custom_app_name = runtime_app_name
            st.rerun()
        if runtime_organization != current_organization:
            st.session_state.custom_organization = runtime_organization
            st.rerun()
        
        # Add button to save settings permanently
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Permanently", help="Save current settings to .env file"):
                save_success = save_settings_to_env(
                    st.session_state.get('custom_app_name', app_name),
                    st.session_state.get('custom_organization', organization_name)
                )
                if save_success:
                    st.success("✅ Settings saved to .env file!")
                    st.info("Restart the app to see changes in page title.")
                else:
                    st.error("❌ Failed to save settings.")
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                if 'custom_app_name' in st.session_state:
                    del st.session_state.custom_app_name
                if 'custom_organization' in st.session_state:
                    del st.session_state.custom_organization
                st.rerun()
        
        st.markdown("---")
    
    st.markdown("### Settings")
    model = st.selectbox("Model", ["gpt-5-mini", "gpt-5"], index=0)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.")
    
    # Only show RAG settings for admins
    if st.session_state.get('is_admin', False):
        st.markdown("---")
        st.markdown("### RAG")
        embedding_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0,
                                       help="Small = cheaper; Large = highest quality")
        persist_dir = st.text_input("Vector store dir", value=".chroma")

# Set default values for variables that might not be defined if sidebar is hidden
if not st.session_state.get('is_admin', False):
    # For public users, set default values
    if 'model' not in locals():
        model = "gpt-5-mini"
    if 'system_prompt' not in locals():
        system_prompt = "You are a helpful assistant."
    embedding_model = "text-embedding-3-small"
    persist_dir = ".chroma"

# Init vector store once
if "chroma_ready" not in st.session_state:
    st.session_state.chroma = get_chroma(persist_dir=persist_dir)
    st.session_state.col = get_collection(st.session_state.chroma, name="docs", openai_api_key=api_key, embedding_model=embedding_model)
    st.session_state.chroma_ready = True

# Upload & index - Only for admins
if st.session_state.get('is_admin', False):
    st.subheader(f"📄 Knowledge Base for {display_organization}")

    # Add input method selection
    input_method = st.radio("Choose input method:", ["Upload Files", "Add URL"])

    if input_method == "Upload Files":
        uploads = st.file_uploader("Add PDFs/TXT/MD", type=["pdf","txt","md"], accept_multiple_files=True)
        if uploads and st.button("Index uploaded files"):
            os.makedirs("tmp_uploads", exist_ok=True)
            saved = []
            for uf in uploads:
                path = os.path.join("tmp_uploads", uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())
                saved.append(path)

            # tip: batch_size 32–64 is gentle on RAM
            n_chunks = ingest_files_streaming(
                saved,
                st.session_state.col,
                chunk_size=1000,        # slightly smaller chunks help
                chunk_overlap=150,
                batch_size=64,
                precompute_embeddings=False,  # start False (Chroma does it); set True if you want tighter control
                openai_client=client,   # only needed if precompute_embeddings=True
                max_file_mb=200
            )
            st.success(f"Ingested {len(saved)} file(s), {n_chunks} chunk(s).")

    else:  # URL input
        url = st.text_input("Enter URL to scrape:")
        
        # Add options for crawling
        col1, col2 = st.columns(2)
        with col1:
            crawl_links = st.checkbox("Crawl all links on page", value=False, help="Index content from all clickable links found on the page")
        with col2:
            max_links = st.number_input("Max links to crawl", min_value=1, max_value=50, value=10, help="Limit number of links to prevent overload")
        
        if url and st.button("Index URL content"):
            try:
                with st.spinner("Fetching content from URL..."):
                    all_content = []
                    processed_urls = set()
                    
                    def extract_content(target_url, is_main=False):
                        try:
                            response = requests.get(target_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                            response.raise_for_status()
                            
                            soup = BeautifulSoup(response.content, 'html.parser')
                            
                            # Extract text content
                            for script in soup(["script", "style", "nav", "footer", "header"]):
                                script.decompose()
                            text_content = soup.get_text()
                            
                            # Clean up whitespace
                            lines = (line.strip() for line in text_content.splitlines())
                            clean_content = '\n'.join(line for line in lines if line)
                            
                            if clean_content.strip():
                                return {
                                    'url': target_url,
                                    'content': clean_content,
                                    'title': soup.title.string if soup.title else 'No title'
                                }
                        except Exception as e:
                            st.warning(f"Failed to process {target_url}: {str(e)}")
                        return None
                    
                    # Extract main page content
                    main_content = extract_content(url, is_main=True)
                    if main_content:
                        all_content.append(main_content)
                        processed_urls.add(url)
                    
                    # If crawling links is enabled
                    if crawl_links and main_content:
                        st.info(f"Crawling links from {url}...")
                        
                        # Get all links from the main page
                        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find all links
                        links = []
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            
                            # Convert relative URLs to absolute
                            if href.startswith('/'):
                                href = f"{url.rstrip('/')}{href}"
                            elif href.startswith('http'):
                                pass  # Already absolute
                            else:
                                continue  # Skip other types
                            
                            # Filter out certain file types and fragments
                            if any(href.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc']):
                                continue
                            if '#' in href:
                                href = href.split('#')[0]  # Remove fragments
                            
                            # Only process same domain links
                            from urllib.parse import urlparse
                            main_domain = urlparse(url).netloc
                            link_domain = urlparse(href).netloc
                            
                            if link_domain == main_domain and href not in processed_urls:
                                links.append(href)
                        
                        # Limit number of links
                        links = links[:max_links]
                        
                        # Process each link
                        progress_bar = st.progress(0)
                        for i, link in enumerate(links):
                            if link not in processed_urls:
                                st.caption(f"Processing: {link}")
                                content = extract_content(link)
                                if content:
                                    all_content.append(content)
                                processed_urls.add(link)
                            
                            progress_bar.progress((i + 1) / len(links))
                    
                    # Save all content to files and ingest
                    if all_content:
                        os.makedirs("tmp_uploads", exist_ok=True)
                        saved_files = []
                        
                        for i, content_item in enumerate(all_content):
                            filename = f"crawled_content_{abs(hash(content_item['url']))}.txt"
                            temp_file = os.path.join("tmp_uploads", filename)
                            
                            with open(temp_file, "w", encoding="utf-8") as f:
                                f.write(f"Source URL: {content_item['url']}\n")
                                f.write(f"Title: {content_item['title']}\n\n")
                                f.write(content_item['content'])
                            
                            saved_files.append(temp_file)
                        
                        # Ingest all content
                        n_chunks = ingest_files_streaming(
                            saved_files,
                            st.session_state.col,
                            chunk_size=1000,
                            chunk_overlap=150,
                            batch_size=64,
                            precompute_embeddings=False,
                            openai_client=client,
                            max_file_mb=200
                        )
                        
                        st.success(f"✅ Crawled {len(all_content)} pages, indexed {n_chunks} chunks.")
                        
                        # Show summary
                        with st.expander("📋 Crawled URLs"):
                            for content_item in all_content:
                                st.write(f"• {content_item['title']} - {content_item['url']}")
                    else:
                        st.error("No content found to index.")
                        
            except requests.RequestException as e:
                st.error(f"Error fetching URL: {str(e)}")
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
                st.stop()
    
    st.caption("Upload files or add a URL to build your knowledge base.")
    st.caption("Tip: index once, then your documents persist in `.chroma/` between app restarts.")
else:
    # For public users, show a message about the knowledge base
    st.info("📚 Knowledge base is available for your queries. Contact admin to add more documents.")
    
    # Add button to show sidebar for public users if needed
    if not st.session_state.get('show_sidebar_public', False):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("⚙️ Show Settings"):
                st.session_state.show_sidebar_public = True
                st.rerun()

# Message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Render prior messages (skip system)
for m in st.session_state.messages:
    if m["role"] == "system": 
        continue
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("Ask something…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # --- RAG step: retrieve & answer ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking with your documents…"):
            top = retrieve(user_msg, st.session_state.col, k=5)
            ctx = build_context_snippets(top, max_chars=3000)
            answer = answer_with_context(client, model, system_prompt, user_msg, ctx)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.rerun()
