import streamlit as st
import pandas as pd
import json
import re
import time
import logging
from typing import List, Dict, Optional
from serpapi import GoogleSearch
from openai import OpenAI
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from logging.handlers import RotatingFileHandler
import os
import sys
def setup_logging(
    log_file='dashboard.log',
    console_level=logging.INFO,
    file_level=logging.DEBUG
):
    """
    Set up advanced logging configuration for the dashboard application.
    
    Args:
        log_file (str): Path to the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Reset any existing handlers
    logging.getLogger().handlers = []
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.error(f"Failed to set up file logging: {e}")
    return logger
def load_google_sheet(sheet_url: str, credentials_json: str) -> pd.DataFrame:
    """
    Load data from Google Sheets using provided credentials
    """
    try:
        # Parse credentials JSON
        credentials_dict = json.loads(credentials_json)
        
        # Define the scope
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        # Authenticate with Google
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        client = gspread.authorize(credentials)
        
        # Extract sheet ID from URL
        sheet_id = re.findall(r'/d/(.*?)/', sheet_url)[0]
        
        # Open the sheet
        sheet = client.open_by_key(sheet_id).sheet1
        
        # Get all values
        data = sheet.get_all_values()
        headers = data[0]
        rows = data[1:]
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=headers)
        return df
        
    except Exception as e:
        raise Exception(f"Error loading Google Sheet: {str(e)}")

def format_results(results: Dict) -> Dict:
    """
    Format the results for export by cleaning and structuring the data
    """
    formatted_results = {
        "metadata": {
            "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_entities": len(results),
            "success_count": sum(1 for r in results.values() if r["success"]),
            "failed_count": sum(1 for r in results.values() if not r["success"])
        },
        "entities": {}
    }
    
    for entity, result in results.items():
        if result["success"]:
            # Parse and structure the results text
            extracted_info = result["results"]
            structured_info = {
                "summary": [],
                "sources": []
            }
            
            # Split into lines and process each
            lines = extracted_info.split('\n')
            current_section = "summary"
            
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith("Source:"):
                        structured_info["sources"].append(line)
                    else:
                        structured_info["summary"].append(line)
            
            formatted_results["entities"][entity] = {
                "status": "success",
                "timestamp": result["timestamp"],
                "search_query": result["search_query"],
                "information": structured_info["summary"],
                "sources": structured_info["sources"]
            }
        else:
            formatted_results["entities"][entity] = {
                "status": "failed",
                "timestamp": result["timestamp"],
                "error": result["error"]
            }
    
    return formatted_results

def create_formatted_csv(results: Dict) -> str:
    """
    Create a formatted CSV string from the results
    """
    formatted_data = []
    
    for entity, data in results["entities"].items():
        row = {
            "Entity": entity,
            "Status": data["status"],
            "Timestamp": data["timestamp"]
        }
        
        if data["status"] == "success":
            row.update({
                "Search Query": data["search_query"],
                "Information": " | ".join(data["information"]),
                "Sources": " | ".join(data["sources"]),
                "Error": ""
            })
        else:
            row.update({
                "Search Query": "",
                "Information": "",
                "Sources": "",
                "Error": data["error"]
            })
            
        formatted_data.append(row)
    
    df = pd.DataFrame(formatted_data)
    return df.to_csv(index=False)
class SearchProcessor:
    def __init__(
        self,
        serpapi_key: str,
        llm_api_key: str,
        llm_type: str = "openai",
        model_name: Optional[str] = None,
        location: str = "United States",
        language: str = "en"
    ):
        """Initialize search processor with API keys and model configuration"""
        if not serpapi_key or not serpapi_key.strip():
            raise ValueError("SerpAPI key is required")
        if not llm_api_key or not llm_api_key.strip():
            raise ValueError(f"{llm_type} API key is required")
        
        try:
            self.serpapi_key = serpapi_key.strip()
            self.location = location
            self.language = language
            
            if llm_type.lower() == "openai":
                if not llm_api_key.startswith(('sk-', 'org-')):
                    raise ValueError("Invalid OpenAI API key format")
                self.client = OpenAI(
                    api_key=llm_api_key.strip(),
                    max_retries=3,
                    timeout=30.0
                )
                self.model_name = model_name or "gpt-4-turbo-preview"
                self.llm_type = "openai"
                
            elif llm_type.lower() == "groq":
                self.client = ChatGroq(
                    api_key=llm_api_key.strip(),
                    temperature=0,
                    model_name=model_name or "mixtral-8x7b-32768",
                    streaming=True
                )
                self.llm_type = "groq"
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SearchProcessor: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: None
    )
    def _search_with_retry(self, query: str) -> Dict:
        """Execute search with retry logic"""
        try:
            params = {
                "q": query,
                "location": self.location,
                "hl": self.language,
                "gl": 'us',
                "google_domain": "google.com",
                "api_key": self.serpapi_key,
                "num": 10  # Get more results for better coverage
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if not results or 'error' in results:
                raise ValueError(f"Invalid search results: {results.get('error', 'Unknown error')}")
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def _prepare_search_query(self, template: str, entity: str) -> str:
        """Prepare search query by replacing entity placeholder with actual value"""
        template = str(template).strip()
        entity = str(entity).strip()
        pattern = re.compile(r'\{entity\}', re.IGNORECASE)
        search_query = pattern.sub(entity, template)
        return search_query

    def _analyze_search_query(self, query: str, entity: str) -> str:
        """Analyze the search query to determine the type of information being requested"""
        try:
            analysis_prompt = f"""Analyze this search query: "{query}" for entity: "{entity}"
            Create focused extraction requirements based on ONLY what is being asked in the query.
            
            Rules:
            1. Only include requirements directly related to the search query
            2. Be specific and concise
            3. Format as a simple list of requirements
            4. Do not add additional requirements not implied by the query
            5. Return ONLY the extraction requirements, no other text
            
            Example 1:
            Query: "What is the capital city of {entity}"
            Requirements:
            1. Capital city name
            2. Source of information
            
            Example 2:
            Query: "Tell about the local specials of {entity}"
            Requirements:
            1. Local food specialties
            2. Notable local dishes
            3. Popular restaurants or venues
            4. Source of information"""
            
            if self.llm_type == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise requirements generator."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0
                )
                return completion.choices[0].message.content
            else:
                return self.client.generate(analysis_prompt)
                
        except Exception as e:
            logger.error(f"Failed to analyze search query: {str(e)}")
            return """Requirements:
            1. Basic information related to the query
            2. Source of information"""

    def _create_extraction_prompt(self, search_results: Dict, entity: str, user_prompt: str) -> str:
        """Create a structured extraction prompt with dynamically generated requirements and explicit source attribution"""
        try:
            # Get dynamic extraction requirements based on the search query
            extraction_requirements = self._analyze_search_query(user_prompt, entity)
            
            # Extract relevant information from search results
            organic_results = search_results.get('organic_results', [])
            knowledge_graph = search_results.get('knowledge_graph', {})
            
            # Combine relevant search data with explicit source attribution
            search_data = {
                "entity": entity,
                "knowledge_graph": knowledge_graph,
                "top_results": [
                    {
                        "title": result.get('title', ''),
                        "snippet": result.get('snippet', ''),
                        "link": result.get('link', ''),
                        "source": "Google Search Result",
                        "position": f"Result #{idx + 1}"
                    }
                    for idx, result in enumerate(organic_results[:5])
                ]
            }

            # Create structured prompt with emphasis on source attribution
            prompt = f"""
            Entity: {entity}
            
            Extraction Requirements:
            {extraction_requirements}
            
            Search Context:
            {json.dumps(search_data, indent=2)}
            
            Instructions:
            1. Extract ONLY the specific information requested in the requirements above
            2. Format the response as clear, simple text (not JSON)
            3. Include only relevant information for {entity}
            4. For EACH piece of information, cite the source using one of these formats:
            - "Source: Google Search Result #X" (where X is the result number)
            - "Source: Google Knowledge Graph"
            5. If information is found in multiple sources, cite all relevant sources
            6. Keep responses focused and concise
            7. IMPORTANT: Always include source attribution even if it's from Google Search
            
            Please provide the extracted information with sources:
            """
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to create extraction prompt: {str(e)}")
            raise

    def _process_with_openai(self, search_results: Dict, entity: str, extraction_prompt: str) -> str:
        """Process search results using OpenAI's API with improved source attribution"""
        try:
            prompt = self._create_extraction_prompt(search_results, entity, extraction_prompt)
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a precise data extraction assistant. Extract only the requested information about {entity}. 
                        Always include source attribution for each piece of information, even if it's from Google Search.
                        Use format 'Source: Google Search Result #X' or 'Source: Google Knowledge Graph'."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            if not completion.choices:
                raise ValueError("No completion choices returned from OpenAI")
                
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI processing failed: {str(e)}")
            raise

    def _process_with_groq(self, search_results: Dict, entity: str, extraction_prompt: str) -> str:
        """Process search results using Groq with improved source attribution"""
        try:
            prompt = self._create_extraction_prompt(search_results, entity, extraction_prompt)
            
            response = self.client.generate(
                f"""As a precise data extraction assistant, extract only the requested information about {entity}.
                Always include source attribution for each piece of information, even if it's from Google Search.
                Use format 'Source: Google Search Result #X' or 'Source: Google Knowledge Graph'.
                
                {prompt}"""
            )
            
            if not response:
                raise ValueError("No response returned from Groq")
                
            return response
            
        except Exception as e:
            logger.error(f"Groq processing failed: {str(e)}")
            raise
    def search_and_extract(
        self,
        entity: str,
        search_template: str,
        extraction_prompt: str
    ) -> Dict:
        """Perform search and extraction for an entity"""
        try:
            if not entity or not search_template:
                raise ValueError("Entity and search template are required")
                
            # Prepare search query
            search_query = self._prepare_search_query(search_template, entity)
            logger.info(f"Prepared search query: {search_query}")
            
            # Perform search
            search_results = self._search_with_retry(search_query)
            
            # Process results with selected LLM
            if self.llm_type == "openai":
                extracted_info = self._process_with_openai(
                    search_results,
                    entity,
                    extraction_prompt or search_template  # Use search template if no extraction prompt
                )
            else:
                extracted_info = self._process_with_groq(
                    search_results,
                    entity,
                    extraction_prompt or search_template
                )
            
            return {
                "success": True,
                "entity": entity,
                "search_query": search_query,
                "results": extracted_info,
                "raw_search": search_results,
                "error": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
                
        except Exception as e:
            logger.error(f"Search and extraction failed for {entity}: {str(e)}")
            return {
                "success": False,
                "entity": entity,
                "search_query": None,
                "results": None,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

def main():
    st.set_page_config(page_title="Data Extraction Dashboard", layout="wide")
    st.title("Data Extraction Dashboard")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = {}
    if 'api_config' not in st.session_state:
        st.session_state.api_config = {
            'serpapi_key': '',
            'llm_type': 'OpenAI',
            'llm_api_key': '',
            'model_name': 'gpt-4-turbo-preview',
            'location': 'United States',
            'language': 'en'
        }

    # Create tabs
    tabs = st.tabs(["Configuration", "Data Upload", "Search & Extract", "Results"])

    with tabs[0]:
        st.header("API Configuration")
        
        with st.form("api_config_form"):
            serpapi_key = st.text_input(
                "SerpAPI Key",
                type="password"
            )
            
            llm_type = st.selectbox(
                "Select LLM Provider",
                ["OpenAI", "Groq"]
            )
            
            llm_api_key = st.text_input(
                f"{llm_type} API Key",
                type="password"
            )
            
            if llm_type == "OpenAI":
                model_name = st.selectbox(
                    "Select OpenAI Model",
                    ["gpt-4-turbo-preview", "gpt-4o-mini", "gpt-3.5-turbo"]
                )
            else:
                model_name = st.selectbox(
                    "Select Groq Model",
                    ["mixtral-8x7b-32768", "llama2-70b-4096"]
                )
            
            location = st.text_input(
                "Search Location",
                value="United States"
            )
            
            language = st.selectbox(
                "Search Language",
                ["en", "es", "fr", "de", "it"]
            )
            
            submitted = st.form_submit_button("Save Configuration")
            
            if submitted:
                try:
                    processor = SearchProcessor(
                        serpapi_key=serpapi_key,
                        llm_api_key=llm_api_key,
                        llm_type=llm_type.lower(),
                        model_name=model_name,
                        location=location,
                        language=language
                    )
                    
                    st.session_state.api_config = {
                        'serpapi_key': serpapi_key,
                        'llm_type': llm_type,
                        'llm_api_key': llm_api_key,
                        'model_name': model_name,
                        'location': location,
                        'language': language
                    }
                    st.success("✅ API configuration validated and saved successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Configuration error: {str(e)}")

    with tabs[1]:
        st.header("Data Upload")
        
        upload_type = st.radio(
            "Select Upload Method",
            ["CSV File", "Google Sheets"]
        )
        
        if upload_type == "CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if df.empty:
                        st.error("The uploaded file is empty")
                    else:
                        st.session_state.df = df
                        st.success("✅ File uploaded successfully!")
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
        
        else:  # Google Sheets
            st.markdown("""
            ### Google Sheets Setup
            1. Share your sheet with the service account email
            2. Paste your Google Sheet URL
            3. Upload your service account credentials JSON file
            """)
            
            sheet_url = st.text_input("Google Sheet URL")
            credentials_file = st.file_uploader("Upload Service Account Credentials", type="json")
            
            if sheet_url and credentials_file:
                try:
                    credentials_json = credentials_file.getvalue().decode()
                    df = load_google_sheet(sheet_url, credentials_json)
                    if df.empty:
                        st.error("The Google Sheet is empty")
                    else:
                        st.session_state.df = df
                        st.success("✅ Google Sheet loaded successfully!")
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading Google Sheet: {str(e)}")

    with tabs[2]:
        st.header("Search & Extract")
        
        if not st.session_state.api_config:
            st.warning("⚠️ Please configure API settings first")
            return
            
        if st.session_state.df is None:
            st.warning("⚠️ Please upload a CSV file first")
            return
            
        main_column = st.selectbox(
            "Select Entity Column",
            options=st.session_state.df.columns.tolist()
        )
        
        st.info("""
        Use {entity} in your queries to reference the selected column value.
        Examples:
        - Search query: "What is the capital city of {entity}"
        - Extraction: "Find the capital city and population of {entity}"
        """)
        
        search_query = st.text_area(
            "Enter search query template",
            "Tell me about {entity}"
        )
        
        if main_column and st.session_state.df is not None:
            sample_entity = st.session_state.df[main_column].iloc[0]
            st.write("Preview for first entity:")
            preview_query = search_query.replace("{entity}", str(sample_entity))
            st.code(preview_query)
            
            # Initialize SearchProcessor to get dynamic extraction requirements
            try:
                config = st.session_state.api_config
                processor = SearchProcessor(
                    serpapi_key=config['serpapi_key'],
                    llm_api_key=config['llm_api_key'],
                    llm_type=config['llm_type'].lower(),
                    model_name=config['model_name'],
                    location=config['location'],
                    language=config['language']
                )
                
                # Get dynamic extraction requirements
                extraction_requirements = processor._analyze_search_query(
                    query=search_query,
                    entity=sample_entity
                )
                
                st.subheader("Generated Extraction Requirements")
                st.info("These requirements are automatically generated based on your search query:")
                st.markdown(extraction_requirements)
                
            except Exception as e:
                st.error(f"Error generating extraction requirements: {str(e)}")
        
        sample_size = st.number_input(
            "Number of entities to process",
            min_value=1,
            max_value=len(st.session_state.df),
            value=min(5, len(st.session_state.df))
        )
        
        if st.button("Start Search & Extract"):
            try:
                config = st.session_state.api_config
                processor = SearchProcessor(
                    serpapi_key=config['serpapi_key'],
                    llm_api_key=config['llm_api_key'],
                    llm_type=config['llm_type'].lower(),
                    model_name=config['model_name'],
                    location=config['location'],
                    language=config['language']
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                sample_df = st.session_state.df.sample(n=sample_size)
                total = len(sample_df)
                
                for idx, (_, row) in enumerate(sample_df.iterrows()):
                    entity = str(row[main_column])
                    status_text.text(f"Processing: {entity} ({idx + 1}/{total})")
                    
                    # Get dynamic extraction requirements for each entity
                    extraction_requirements = processor._analyze_search_query(
                        query=search_query,
                        entity=entity
                    )
                    
                    result = processor.search_and_extract(
                        entity=entity,
                        search_template=search_query,
                        extraction_prompt=extraction_requirements
                    )
                    
                    st.session_state.search_results[entity] = result
                    progress_bar.progress((idx + 1) / total)
                    
                    # Display results in a container
                    with st.container():
                        st.markdown(f"### {entity}")
                        if result['success']:
                            st.markdown(result['results'])
                        else:
                            st.error(f"Error: {result['error']}")
                        st.divider()  # Add visual separation between entities
                
                status_text.text("✅ Processing complete!")
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

    with tabs[3]:
        st.header("Results")
        if st.session_state.search_results:
            # Add filters
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=["Successful", "Failed"],
                    default=["Successful", "Failed"]
                )
            
            # Filter results based on selection
            filtered_results = {}
            for entity, result in st.session_state.search_results.items():
                if (result["success"] and "Successful" in status_filter) or \
                   (not result["success"] and "Failed" in status_filter):
                    filtered_results[entity] = result
            
            if filtered_results:
                selected_entity = st.selectbox(
                    "Select Entity",
                    options=list(filtered_results.keys())
                )
                
                if selected_entity:
                    result = filtered_results[selected_entity]
                    
                    # Display result details
                    result_cols = st.columns(3)
                    with result_cols[0]:
                        st.metric("Status", "✅ Success" if result["success"] else "❌ Failed")
                    with result_cols[1]:
                        st.metric("Timestamp", result["timestamp"])
                    with result_cols[2]:
                        if result.get("search_query"):
                            st.metric("Search Query Used", result["search_query"])
                    
                    # Create tabs for different views of the results
                    result_tabs = st.tabs(["Processed Results", "Raw Search Results"])
                    
                    with result_tabs[0]:
                        if result["success"]:
                            st.markdown("### Extracted Information")
                            st.markdown(result["results"])
                        else:
                            st.error(f"Processing failed: {result['error']}")
                    
                    with result_tabs[1]:
                        if result.get("raw_search"):
                            st.markdown("### Raw Search Results")
                            st.json(result["raw_search"])
                # Export functionality
                st.markdown("---")
                st.markdown("### Export Options")
                export_cols = st.columns(2)
                
                with export_cols[0]:
                    if st.button("Export Results (JSON)"):
                        formatted_results = format_results(filtered_results)
                        json_str = json.dumps(formatted_results, indent=2)
                        st.download_button(
                            "📥 Download Formatted JSON",
                            data=json_str,
                            file_name=f"formatted_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with export_cols[1]:
                    if st.button("Export Results (CSV)"):
                        formatted_results = format_results(filtered_results)
                        csv_str = create_formatted_csv(formatted_results)
                        st.download_button(
                            "📥 Download Formatted CSV",
                            data=csv_str,
                            file_name=f"formatted_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                # Clear results option
                st.markdown("---")
                if st.button("Clear All Results"):
                    confirm = st.checkbox("Are you sure you want to clear all results?")
                    if confirm:
                        st.session_state.search_results = {}
                        st.success("Results cleared successfully!")
                        st.experimental_rerun()
            else:
                st.info("No results match the selected filters.")
        else:
            st.info("No results available yet. Please run the search and extraction process first.")

if __name__ == "__main__":
    try:
        logger = setup_logging()  # Initialize logging
        logger.info("Starting Data Extraction Dashboard")
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")