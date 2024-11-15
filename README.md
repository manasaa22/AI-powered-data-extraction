# Data Extraction Dashboard

A powerful Streamlit-based dashboard for automated data extraction and analysis using Google Search, LLMs (OpenAI/Groq), and flexible data input methods.

## 🌟 Features

- **Multiple Data Input Methods**
  - CSV file upload
  - Google Sheets integration
  - Real-time data preview

- **Flexible Search Configuration**
  - Dynamic query templating with {entity} placeholder
  - Location-based search customization
  - Multi-language support

- **Advanced LLM Processing**
  - OpenAI GPT-40-mini and GPT-3.5 integration
  - Groq API support with Mixtral and LLaMA models
  - Dynamic extraction requirement generation

- **Comprehensive Results Management**
  - Real-time progress tracking
  - Filterable results view
  - Export options (JSON/CSV)
  - Detailed logging system

## 🚀 Setup Instructions

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/manasaa22/AI-powered-data-extraction.git
cd AI-powered-data-extraction

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

Create a `requirements.txt` file with the following packages:

```
streamlit
pandas
serpapi
openai
langchain-groq
tenacity
gspread
oauth2client
```

## 🎯 Usage Guide

### 1. Configuration Tab

1. Enter your API keys:
   - SerpAPI key for Google Search
   - OpenAI or Groq API key for LLM processing
2. Select LLM provider and model
3. Configure search location and language
4. Save configuration

### 2. Data Upload Tab

#### CSV Upload
1. Select "CSV File" option
2. Upload your CSV file containing entities to process
3. Verify data preview

#### Google Sheets Integration
1. Select "Google Sheets" option
2. Set up Google Cloud Project and enable Sheets API
3. Create service account and download credentials
4. Share your sheet with the service account email
5. Enter sheet URL and upload credentials JSON
6. Verify data preview

### 3. Search & Extract Tab

1. Select entity column from your dataset
2. Enter search query template using {entity} placeholder
   Example: "What is the capital city of {entity}"
3. Review generated extraction requirements
4. Set number of entities to process
5. Start extraction process

### 4. Results Tab

1. View processed results with filtering options
2. Examine detailed information for each entity
3. Access raw search results
4. Export results in JSON or CSV format
5. Clear results if needed

## 🔑 API Keys Setup

### SerpAPI
1. Visit [SerpAPI](https://serpapi.com/)
2. Create account and get API key
3. Enter key in dashboard configuration

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create account and generate API key
3. Enter key in dashboard configuration

### Groq (Optional)
1. Visit [Groq](https://console.groq.com/)
2. Create account and get API key
3. Enter key in dashboard configuration

### Google Sheets Integration
1. Create project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Sheets API
3. Create service account and download credentials
4. Share target sheets with service account email

## 📝 Optional Features

1. **Advanced Logging System**
   - Detailed error tracking
   - Rotating file handler
   - Console and file output

2. **Dynamic Extraction Requirements**
   - AI-powered requirement generation
   - Context-aware information extraction

3. **Resilient Processing**
   - Automatic retries with exponential backoff
   - Error handling and recovery

4. **Export Flexibility**
   - Formatted JSON export
   - CSV export with structured data

## 📈 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.