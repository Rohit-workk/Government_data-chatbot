import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import time
from groq_client import GroqClient
from io import StringIO
import sys

# ==== Console Output Capture ====
class StreamCapture:
    def __init__(self):
        self.output = StringIO()
        self.original_stdout = sys.stdout
    
    def start(self):
        sys.stdout = self.output
    
    def stop(self):
        sys.stdout = self.original_stdout
        return self.output.getvalue()

# ==== DataGovIndia API Client ====
class DataGovIndia:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.data.gov.in/resource"
    
    def get_data(self, resource_id, filters=None, fields=None, limit=1000):
        params = {"api-key": self.api_key, "format": "json", "limit": limit}
        if filters:
            for key, value in filters.items():
                params[f"filters[{key}]"] = value
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            url = f"{self.base_url}/{resource_id}"
            print(f"🔍 Fetching: {url}")
            print(f"   Filters: {filters}")
            
            response = requests.get(url, params=params, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   ❌ Error: {response.text[:200]}")
                return None
            
            data = response.json()
            
            if "records" in data:
                df = pd.DataFrame(data["records"])
                print(f"   ✅ Fetched {len(df)} records")
                return df
            elif isinstance(data, list):
                df = pd.DataFrame(data)
                print(f"   ✅ Fetched {len(df)} records")
                return df
            else:
                return pd.DataFrame(data) if data else None
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return None

# ==== Helper Functions ====
def safe_display_dataframe(df, max_rows=100):
    if df is None or df.empty:
        return pd.DataFrame()
    
    display_df = df.head(max_rows).copy()
    
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].astype(str)
        display_df[col] = display_df[col].replace(
            [None, np.nan, 'nan', 'NaN', 'None', 'none', ''], 'NA'
        )
    
    return display_df

def estimate_tokens(text):
    """Rough token estimation"""
    return len(text) // 4

def detect_query_intent(query_text):
    """Detect query intent from keywords"""
    query_lower = query_text.lower()
    
    intent_keywords = {
        "comparison": ["compare", "vs", "versus", "difference between", "between"],
        "trend": ["trend", "over time", "historical", "change", "decade", "years"],
        "top_list": ["top", "highest", "lowest", "best", "worst", "most", "least"],
        "correlation": ["correlate", "relationship", "impact", "affect", "influence"],
        "district": ["district", "tehsil", "block", "taluka"],
        "policy": ["policy", "scheme", "recommendation", "suggest", "propose"]
    }
    
    detected_intents = []
    for intent, keywords in intent_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_intents.append(intent)
    
    return detected_intents[0] if detected_intents else "general"

def get_fields_list(fields_data):
    """Extract field names"""
    if pd.isna(fields_data) or fields_data == '' or str(fields_data).lower() in ['nan', 'none']:
        return []
    
    try:
        if isinstance(fields_data, str):
            fields = json.loads(fields_data)
        else:
            fields = fields_data
        
        if isinstance(fields, list):
            field_names = []
            for field in fields:
                if isinstance(field, dict):
                    field_names.append(field.get('name', field.get('id', '')))
                elif isinstance(field, str):
                    field_names.append(field)
            return field_names
        return []
    except:
        return []

def check_required_fields(dataset_fields, required_fields):
    """Check if dataset has required fields"""
    if not required_fields:
        return True
    
    dataset_fields_lower = [f.lower() for f in dataset_fields]
    
    for req_field in required_fields:
        # Check for exact or partial match
        if any(req_field.lower() in df for df in dataset_fields_lower):
            continue
        else:
            return False
    
    return True

# ==== Streamlit Configuration ====
st.set_page_config(
    page_title="Gov Data Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Load Metadata ====
@st.cache_data
def load_metadata():
    df_imd = pd.read_csv('data/imd_eng_feature_df.csv')
    df_agriculture = pd.read_csv('data/Agr_eng_feature_dfi.csv')
    df_imd['source'] = 'imd'
    df_agriculture['source'] = 'agriculture'
    return df_imd, df_agriculture

df_imd, df_agriculture = load_metadata()

IMD_CATEGORIES = sorted(df_imd['category'].unique().tolist()) if 'category' in df_imd.columns else []
AGRICULTURE_CATEGORIES = sorted(df_agriculture['category'].unique().tolist()) if 'category' in df_agriculture.columns else []

# ==== Initialize Session State ====
if "messages" not in st.session_state:
    st.session_state.messages = []

if "groq_client" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    st.session_state.groq_client = GroqClient(api_key=api_key)

if "govdata" not in st.session_state:
    api_key = st.secrets.get("DATAGOV_API_KEY", "")
    st.session_state.govdata = DataGovIndia(api_key=api_key)

if "console_output" not in st.session_state:
    st.session_state.console_output = ""

if "filtered_datasets" not in st.session_state:
    st.session_state.filtered_datasets = None

if "waiting_for_selection" not in st.session_state:
    st.session_state.waiting_for_selection = False

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

if "current_routing" not in st.session_state:
    st.session_state.current_routing = None

if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ==== Core Functions ====

def analyze_query_and_route(user_query, client):
    """LLM Call 1: STRICT routing with keyword detection"""
    try:
        # Detect intent first
        detected_intent = detect_query_intent(user_query)
        
        # Keyword-based strict routing
        query_lower = user_query.lower()
        
        # Agriculture keywords
        agri_keywords = ["crop", "agriculture", "production", "farming", "harvest", 
                        "pesticide", "fertilizer", "yield", "cultivation", "sowing",
                        "rice", "wheat", "cotton", "sugarcane", "maize"]
        
        # Weather keywords
        weather_keywords = ["rainfall", "weather", "storm", "cyclone", "temperature",
                           "climate", "precipitation", "monsoon", "drought"]
        
        has_agri = any(kw in query_lower for kw in agri_keywords)
        has_weather = any(kw in query_lower for kw in weather_keywords)
        
        # STRICT routing logic
        if has_agri and not has_weather:
            forced_source = ["agriculture"]
        elif has_weather and not has_agri:
            forced_source = ["imd"]
        else:
            forced_source = ["both"]
        
        prompt = f"""Analyze: "{user_query}"

States: Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, West Bengal, Delhi, Jammu & Kashmir, Ladakh

IMD Categories: {', '.join(IMD_CATEGORIES)}
Agriculture Categories: {', '.join(AGRICULTURE_CATEGORIES)}

Extract ALL:
- states: ALL states (expand "multiple" to actual names)
- districts: District names
- years: ALL years as integers (expand "2015-2020" → [2015,2016,2017,2018,2019,2020])
- months: lowercase
- metrics: EXACT category names
- crops: ALL crop names
- required_fields: Fields needed (e.g., ["production", "yield", "rainfall"])

Return JSON:
{{"sources_needed": {json.dumps(forced_source)}, "entities": {{"states": [], "districts": [], "years": [], "months": [], "metrics": [], "crops": [], "required_fields": [], "intent": "{detected_intent}", "numbers": {{}}}}}}"""
        
        tokens_in = estimate_tokens(prompt)
        
        response = client.chat_completions_create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600
        )
        
        text = response.choices[0].message.content.strip()
        tokens_out = estimate_tokens(text)
        
        st.session_state.tokens_used += (tokens_in + tokens_out)
        
        if text.startswith("```"):
            text = text.split("```")[1].replace("json", "").strip()
        
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1:
            text = text[json_start:json_end]
        
        routing = json.loads(text)
        
        # Override with forced source
        routing['sources_needed'] = forced_source
        routing['entities']['intent'] = detected_intent
        
        print(f"✅ Routing: {routing.get('sources_needed')}")
        print(f"   Intent: {detected_intent}")
        print(f"   Entities: {routing.get('entities')}")
        
        return routing
        
    except Exception as e:
        print(f"⚠️ Routing error: {e}")
        return {
            "sources_needed": ["both"],
            "entities": {"states": [], "districts": [], "years": [], "months": [], "metrics": [], "crops": [], "required_fields": [], "intent": "general", "numbers": {}}
        }

def filter_imd_datasets(df_imd, entities, top_n=10):
    """IMPROVED IMD filtering with strict scoring"""
    df = df_imd.copy()
    scores = []
    
    required_fields = entities.get('required_fields', [])
    intent = entities.get('intent', 'general')
    
    for _, row in df.iterrows():
        score = 0
        
        # PRIORITY 1: Exact category match (HIGHEST)
        if entities.get('metrics'):
            category = str(row.get('category', '')).lower()
            title = str(row.get('title', '')).lower()
            
            for metric in entities['metrics']:
                metric_lower = metric.lower()
                # Exact match
                if metric_lower == category:
                    score += 200  # Doubled from 100
                # Partial match
                elif metric_lower in category or category in metric_lower:
                    score += 50
                # Title match
                if metric_lower in title:
                    score += 20
        
        # PRIORITY 2: Multi-entity bonus (state + year + metric ALL match)
        has_state = False
        has_year = False
        has_metric = False
        
        if entities.get('states'):
            states_field = str(row.get('states_mentioned', '')).lower()
            for state in entities['states']:
                if state.lower() in states_field:
                    has_state = True
                    score += 30  # Increased from 20
                    break
        
        if entities.get('years'):
            try:
                if pd.notna(row.get('start_year')) and pd.notna(row.get('end_year')):
                    query_years = set(entities['years'])
                    dataset_years = set(range(int(row['start_year']), int(row['end_year']) + 1))
                    overlap = query_years.intersection(dataset_years)
                    if overlap:
                        has_year = True
                        # Proportional scoring
                        overlap_percent = len(overlap) / len(query_years)
                        score += int(40 * overlap_percent)  # Up to 40 points
            except:
                pass
        
        if entities.get('metrics'):
            has_metric = True
        
        # ALL THREE MATCH BONUS
        if has_state and has_year and has_metric:
            score += 100  # HUGE BONUS
        
        # PRIORITY 3: Required fields check
        dataset_fields = get_fields_list(row.get('fields', ''))
        if required_fields:
            if check_required_fields(dataset_fields, required_fields):
                score += 50  # Field availability bonus
            else:
                score -= 50  # Penalty for missing fields
        
        # PRIORITY 4: Intent matching
        if intent == "trend":
            # Prefer datasets with longer time periods
            try:
                if pd.notna(row.get('start_year')) and pd.notna(row.get('end_year')):
                    time_span = int(row['end_year']) - int(row['start_year'])
                    if time_span >= 5:
                        score += 30
            except:
                pass
        elif intent == "comparison":
            # Prefer datasets mentioning multiple states
            states_mentioned = str(row.get('states_mentioned', '')).lower()
            if 'all' in states_mentioned or 'india' in states_mentioned:
                score += 25
        
        scores.append(score)
    
    df['relevance_score'] = scores
    result = df[df['relevance_score'] > 0].nlargest(top_n, 'relevance_score')
    
    if result.empty:
        print(f"⚠️ No IMD matches, using fallback")
        result = df.head(top_n)
        result['relevance_score'] = 0
    else:
        print(f"✅ IMD: {len(result)} datasets, top score: {result['relevance_score'].max()}")
    
    return result

def filter_agriculture_datasets(df_agriculture, entities, top_n=10):
    """IMPROVED Agriculture filtering with strict scoring"""
    df = df_agriculture.copy()
    scores = []
    
    required_fields = entities.get('required_fields', [])
    intent = entities.get('intent', 'general')
    
    for _, row in df.iterrows():
        score = 0
        
        # PRIORITY 1: Exact category match
        if entities.get('metrics') or entities.get('crops'):
            category = str(row.get('category', '')).lower()
            title = str(row.get('title', '')).lower()
            
            for metric in entities.get('metrics', []):
                metric_lower = metric.lower()
                if metric_lower == category:
                    score += 200
                elif metric_lower in category or category in metric_lower:
                    score += 50
                if metric_lower in title:
                    score += 20
            
            for crop in entities.get('crops', []):
                crop_lower = crop.lower()
                if crop_lower in title or crop_lower in category:
                    score += 40  # Increased from 30
        
        # PRIORITY 2: Multi-entity bonus
        has_state = False
        has_year = False
        has_crop = False
        
        if entities.get('states'):
            states_field = str(row.get('states_mentioned', '')).lower()
            for state in entities['states']:
                if state.lower() in states_field:
                    has_state = True
                    score += 30
                    break
        
        if entities.get('years'):
            try:
                if pd.notna(row.get('start_year')) and pd.notna(row.get('end_year')):
                    query_years = set(entities['years'])
                    dataset_years = set(range(int(row['start_year']), int(row['end_year']) + 1))
                    overlap = query_years.intersection(dataset_years)
                    if overlap:
                        has_year = True
                        overlap_percent = len(overlap) / len(query_years)
                        score += int(40 * overlap_percent)
            except:
                pass
        
        if entities.get('crops'):
            has_crop = True
        
        # ALL MATCH BONUS
        if has_state and has_year and has_crop:
            score += 100
        
        # PRIORITY 3: Required fields
        dataset_fields = get_fields_list(row.get('fields', ''))
        if required_fields:
            if check_required_fields(dataset_fields, required_fields):
                score += 50
            else:
                score -= 50
        
        # PRIORITY 4: Intent matching
        if intent == "trend":
            try:
                if pd.notna(row.get('start_year')) and pd.notna(row.get('end_year')):
                    time_span = int(row['end_year']) - int(row['start_year'])
                    if time_span >= 5:
                        score += 30
            except:
                pass
        elif intent == "top_list":
            # Prefer datasets with production/yield data
            title_lower = str(row.get('title', '')).lower()
            if 'production' in title_lower or 'yield' in title_lower:
                score += 25
        
        scores.append(score)
    
    df['relevance_score'] = scores
    result = df[df['relevance_score'] > 0].nlargest(top_n, 'relevance_score')
    
    if result.empty:
        print(f"⚠️ No Agriculture matches, using fallback")
        result = df.head(top_n)
        result['relevance_score'] = 0
    else:
        print(f"✅ Agriculture: {len(result)} datasets, top score: {result['relevance_score'].max()}")
    
    return result

def filter_datasets_by_source(routing_info, top_n=10):
    """STRICT routing - blocks wrong sources"""
    sources = routing_info.get('sources_needed', ['both'])
    entities = routing_info.get('entities', {})
    
    print(f"\n🎯 STRICT Routing: {sources}")
    
    results = []
    
    # STRICT: Only filter from allowed sources
    if sources == ['imd']:
        # IMD ONLY - block agriculture completely
        imd_result = filter_imd_datasets(df_imd, entities, top_n)
        if not imd_result.empty:
            results.append(imd_result)
        print("✅ IMD only (Agriculture BLOCKED)")
    
    elif sources == ['agriculture']:
        # AGRICULTURE ONLY - block IMD completely
        agr_result = filter_agriculture_datasets(df_agriculture, entities, top_n)
        if not agr_result.empty:
            results.append(agr_result)
        print("✅ Agriculture only (IMD BLOCKED)")
    
    else:
        # BOTH sources allowed
        imd_result = filter_imd_datasets(df_imd, entities, top_n//2)
        agr_result = filter_agriculture_datasets(df_agriculture, entities, top_n//2)
        if not imd_result.empty:
            results.append(imd_result)
        if not agr_result.empty:
            results.append(agr_result)
        print("✅ Both sources allowed")
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        combined = combined.sort_values('relevance_score', ascending=False)
        print(f"✅ Total: {len(combined)} datasets")
        return combined
    
    print("⚠️ No datasets found")
    return pd.DataFrame()

def generate_fetch_parameters(selected_datasets, entities, client):
    """LLM Call 2: Generate parameters"""
    try:
        datasets_info = []
        for _, row in selected_datasets.iterrows():
            fields = get_fields_list(row.get('fields', ''))
            datasets_info.append({
                "resource_id": row['resource_id'],
                "title": row['title'][:70],
                "category": row.get('category', ''),
                "available_fields": fields[:15]
            })
        
        prompt = f"""Generate API parameters for COMPLEX QUERY.

User Query Entities: {json.dumps(entities)}

Datasets:
{json.dumps(datasets_info, indent=2)}

Generate parameters with EXACT field names.
For multi-state queries, create SEPARATE parameter sets for EACH state.

Return JSON array:
[{{"resource_id": "...", "filters": {{"state": "Maharashtra", "year": "2020"}}, "fields": [], "limit": 5000}}]"""
        
        tokens_in = estimate_tokens(prompt)
        
        response = client.chat_completions_create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1200
        )
        
        text = response.choices[0].message.content.strip()
        tokens_out = estimate_tokens(text)
        
        st.session_state.tokens_used += (tokens_in + tokens_out)
        
        if text.startswith("```"):
            lines = text.split('\n')
            text = '\n'.join([('```') and 'json' not in l.lower()])
        
        json_start = text.find('[')
        json_end = text.rfind(']') + 1
        if json_start != -1:
            text = text[json_start:json_end]
        
        params_list = json.loads(text)
        print(f"✅ Generated {len(params_list)} parameter sets")
        return params_list
        
    except Exception as e:
        print(f"⚠️ Parameter error: {e}")
        return [{"resource_id": row['resource_id'], "filters": {}, "fields": [], "limit": 5000} 
                for _, row in selected_datasets.iterrows()]

def generate_answer_from_data(query, fetched_data, routing_info, client):
    """LLM Call 3: Generate comprehensive answer"""
    try:
        data_summary = []
        for rid, info in fetched_data.items():
            df = info['data']
            data_summary.append({
                "title": info['title'],
                "rows": len(df),
                "columns": list(df.columns)[:12],
                "sample": df.head(4).to_dict('records')
            })
        
        intent = routing_info.get('entities', {}).get('intent', 'general')
        
        prompt = f"""Answer COMPLEX QUERY: "{query}"

Intent: {intent}
Entities: {routing_info.get('entities', {})}

Data ({len(fetched_data)} datasets):
{json.dumps(data_summary, default=str)}

Provide comprehensive answer with:
1. Direct answer with numbers
2. State/district comparisons
3. Trends if applicable
4. Top lists if applicable
5. Correlations if applicable
6. Policy insights if applicable

Max 600 words."""
        
        tokens_in = estimate_tokens(prompt)
        
        response = client.chat_completions_create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        
        text = response.choices[0].message.content
        tokens_out = estimate_tokens(text)
        
        st.session_state.tokens_used += (tokens_in + tokens_out)
        
        return text
        
    except Exception as e:
        print(f"⚠️ Answer error: {e}")
        return f"## Data Fetched ({len(fetched_data)} datasets)\n\nReview raw data below."

# ==== Streamlit UI ====

with st.sidebar:
    st.markdown("# 🤖 Gov Data Chatbot")
    st.markdown("**IMPROVED:** Strict routing + Better filtering")
    st.markdown("---")
    
    st.markdown("### 📊 Data Sources")
    st.info(f"**IMD:** {len(df_imd)}")
    st.info(f"**Agriculture:** {len(df_agriculture)}")
    
    st.markdown("---")
    
    # Token usage
    st.markdown("### 🔢 Token Usage")
    TOKEN_LIMIT = 100000
    tokens_remaining = TOKEN_LIMIT - st.session_state.tokens_used
    usage_percent = (st.session_state.tokens_used / TOKEN_LIMIT) * 100
    
    st.progress(min(usage_percent / 100, 1.0))
    st.metric("Tokens Used", f"{st.session_state.tokens_used:,}")
    st.metric("Remaining", f"{max(tokens_remaining, 0):,}")
    st.metric("Queries", st.session_state.query_count)
    
    if usage_percent > 80:
        st.warning("⚠️ Low tokens!")
    
    st.markdown("---")
    
    with st.expander("📋 IMD Categories"):
        for i, cat in enumerate(IMD_CATEGORIES, 1):
            st.markdown(f"{i}. {cat}")
    
    with st.expander("📋 Agriculture Categories"):
        for i, cat in enumerate(AGRICULTURE_CATEGORIES, 1):
            st.markdown(f"{i}. {cat}")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("checkbox_"):
                del st.session_state[key]
        st.session_state.messages = []
        st.session_state.filtered_datasets = None
        st.session_state.waiting_for_selection = False
        st.session_state.console_output = ""
        st.rerun()

st.title("🤖 Government Data Q&A Chatbot")
st.markdown("**✨ IMPROVED:** Strict source routing • Better filtering • Multi-entity matching")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total", len(df_imd) + len(df_agriculture))
with col2:
    st.metric("IMD", len(df_imd))
with col3:
    st.metric("Agriculture", len(df_agriculture))

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("dataframes"):
            with st.expander("📋 View Raw Data"):
                for rid, info in msg["dataframes"].items():
                    st.write(f"**{info['title']}**")
                    display_df = safe_display_dataframe(info['data'], 100)
                    st.dataframe(display_df, use_container_width=True)
                    
                    csv = info['data'].to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"data_{rid}.csv",
                        "text/csv",
                        key=f"dl_{rid}_{msg.get('timestamp', '')}"
                    )

# Dataset Selection UI
if st.session_state.waiting_for_selection and st.session_state.filtered_datasets is not None:
    st.markdown("### 🎯 Select Datasets")
    
    filtered_df = st.session_state.filtered_datasets
    
    imd_datasets = filtered_df[filtered_df['source'] == 'imd']
    agr_datasets = filtered_df[filtered_df['source'] == 'agriculture']
    
    if not imd_datasets.empty:
        st.markdown("#### 🌦️ IMD Datasets")
        for idx, row in imd_datasets.iterrows():
            rid = row.get('resource_id', '')
            col1, col2 = st.columns([0.1, 0.9])
            
            with col1:
                st.checkbox("", key=f"checkbox_{rid}", value=False)
            
            with col2:
                with st.expander(f"**{row.get('title', '')[:70]}** (Score: {row.get('relevance_score', 0)})"):
                    st.write(f"**Resource ID:** `{rid}`")
                    st.write(f"**Category:** {row.get('category', '')}")
                    
                    fields = get_fields_list(row.get('fields', ''))
                    if fields:
                        st.write(f"**Fields ({len(fields)}):** {', '.join(fields[:10])}")
    
    if not agr_datasets.empty:
        st.markdown("#### 🌾 Agriculture Datasets")
        for idx, row in agr_datasets.iterrows():
            rid = row.get('resource_id', '')
            col1, col2 = st.columns([0.1, 0.9])
            
            with col1:
                st.checkbox("", key=f"checkbox_{rid}", value=False)
            
            with col2:
                with st.expander(f"**{row.get('title', '')[:70]}** (Score: {row.get('relevance_score', 0)})"):
                    st.write(f"**Resource ID:** `{rid}`")
                    st.write(f"**Category:** {row.get('category', '')}")
                    
                    fields = get_fields_list(row.get('fields', ''))
                    if fields:
                        st.write(f"**Fields ({len(fields)}):** {', '.join(fields[:10])}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True):
            selected_ids = [key.replace("checkbox_", "") for key in st.session_state.keys() 
                          if key.startswith("checkbox_") and st.session_state[key]]
            
            if len(selected_ids) > 0:
                with st.spinner(f"🧠 Processing {len(selected_ids)} datasets..."):
                    capture = StreamCapture()
                    capture.start()
                    
                    selected_datasets = filtered_df[filtered_df['resource_id'].isin(selected_ids)]
                    
                    params_list = generate_fetch_parameters(
                        selected_datasets,
                        st.session_state.current_routing.get('entities', {}),
                        st.session_state.groq_client
                    )
                    
                    fetched = {}
                    for param in params_list:
                        rid = param['resource_id']
                        try:
                            data = st.session_state.govdata.get_data(
                                rid,
                                filters=param.get('filters'),
                                fields=param.get('fields'),
                                limit=param.get('limit', 5000)
                            )
                            if data is not None and not data.empty:
                                title = selected_datasets[selected_datasets['resource_id']==rid]['title'].values[0]
                                filter_key = str(param.get('filters', {}))
                                unique_key = f"{rid}_{hash(filter_key)}"
                                fetched[unique_key] = {'title': title, 'data': data}
                                st.success(f"✅ {len(data)} rows")
                        except Exception as e:
                            st.error(f"❌ {e}")
                    
                    console_out = capture.stop()
                    st.session_state.console_output = console_out
                    
                    if fetched:
                        with st.spinner("🧠 Generating answer..."):
                            answer_text = generate_answer_from_data(
                                st.session_state.current_query,
                                fetched,
                                st.session_state.current_routing,
                                st.session_state.groq_client
                            )
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "dataframes": fetched,
                            "timestamp": str(time.time())
                        })
                        
                        for key in list(st.session_state.keys()):
                            if key.startswith("checkbox_"):
                                del st.session_state[key]
                        
                        st.session_state.waiting_for_selection = False
                        st.session_state.filtered_datasets = None
                        st.rerun()
            else:
                st.warning("⚠️ Select at least one dataset")
    
    with col2:
        if st.button("🤖 Auto-Select Balanced", use_container_width=True):
            selected_datasets = pd.DataFrame()
            
            if not imd_datasets.empty and not agr_datasets.empty:
                imd_top = imd_datasets.nlargest(2, 'relevance_score')
                agr_top = agr_datasets.nlargest(2, 'relevance_score')
                selected_datasets = pd.concat([imd_top, agr_top], ignore_index=True)
            elif not imd_datasets.empty:
                selected_datasets = imd_datasets.nlargest(4, 'relevance_score')
            elif not agr_datasets.empty:
                selected_datasets = agr_datasets.nlargest(4, 'relevance_score')
            
            if not selected_datasets.empty:
                with st.spinner("🧠 Processing..."):
                    capture = StreamCapture()
                    capture.start()
                    
                    params_list = generate_fetch_parameters(
                        selected_datasets,
                        st.session_state.current_routing.get('entities', {}),
                        st.session_state.groq_client
                    )
                    
                    fetched = {}
                    for param in params_list:
                        rid = param['resource_id']
                        try:
                            data = st.session_state.govdata.get_data(
                                rid,
                                filters=param.get('filters'),
                                fields=param.get('fields'),
                                limit=param.get('limit', 5000)
                            )
                            if data is not None and not data.empty:
                                title = selected_datasets[selected_datasets['resource_id']==rid]['title'].values[0]
                                filter_key = str(param.get('filters', {}))
                                unique_key = f"{rid}_{hash(filter_key)}"
                                fetched[unique_key] = {'title': title, 'data': data}
                        except Exception as e:
                            print(f"Error: {e}")
                    
                    console_out = capture.stop()
                    st.session_state.console_output = console_out
                    
                    if fetched:
                        answer_text = generate_answer_from_data(
                            st.session_state.current_query,
                            fetched,
                            st.session_state.current_routing,
                            st.session_state.groq_client
                        )
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "dataframes": fetched,
                            "timestamp": str(time.time())
                        })
                        
                        st.session_state.waiting_for_selection = False
                        st.session_state.filtered_datasets = None
                        st.rerun()

# Console Output
if st.session_state.console_output:
    with st.expander("🖥️ Console Output"):
        st.code(st.session_state.console_output, language="text")

# Chat input
if not st.session_state.waiting_for_selection:
    user_input = st.chat_input("Ask complex questions...")
    
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": str(time.time())
        })
        
        st.session_state.query_count += 1
        
        with st.spinner("🧠 Analyzing..."):
            capture = StreamCapture()
            capture.start()
            
            routing = analyze_query_and_route(user_input, st.session_state.groq_client)
            filtered = filter_datasets_by_source(routing, top_n=10)
            
            console_out = capture.stop()
            st.session_state.console_output = console_out
            
            if not filtered.empty:
                st.session_state.filtered_datasets = filtered
                st.session_state.waiting_for_selection = True
                st.session_state.current_query = user_input
                st.session_state.current_routing = routing
                st.rerun()
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ No relevant datasets found",
                    "timestamp": str(time.time())
                })
                st.rerun()

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🚀 Powered by ROHIT | Improved Filtering v2.0</div>", unsafe_allow_html=True)
