# app.py - Enhanced Smart Closet
import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import json
from datetime import datetime
import io
import base64

# -----------------------------------------------------
# Page config
# -----------------------------------------------------
st.set_page_config(
    page_title="Smart Closet",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Session state initialization
# -----------------------------------------------------
if 'wardrobe_items' not in st.session_state:
    st.session_state.wardrobe_items = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'weather': '',
        'event': '',
        'style_preference': [],
        'color_preference': []
    }

# -----------------------------------------------------
# Load background image
# -----------------------------------------------------
def _get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

try:
    import os
    bg_path = os.path.join(os.path.dirname(__file__), "bgphoto.png")
    bg_b64 = _get_base64_of_bin_file(bg_path)
except:
    bg_b64 = None

# -----------------------------------------------------
# Enhanced Fashion Knowledge Base
# -----------------------------------------------------
FASHION_TRENDS = {
    'barbiecore': ['pink', 'feminine', 'playful', 'bright', 'pastel pink', 'hot pink', 'girly'],
    'cottagecore': ['floral', 'vintage', 'pastoral', 'soft', 'prairie dress', 'natural', 'romantic'],
    'dark academia': ['brown', 'beige', 'vintage', 'scholarly', 'tweed', 'plaid', 'bookish'],
    'streetwear': ['oversized', 'sneakers', 'hoodie', 'urban', 'casual', 'athletic', 'sporty'],
    'minimalist': ['simple', 'clean', 'neutral', 'basic', 'monochrome', 'sleek', 'understated'],
    'y2k': ['metallic', 'butterfly', 'low-rise', 'colorful', 'nostalgic', '2000s', 'nostalgic'],
    'coastal grandmother': ['linen', 'relaxed', 'beige', 'coastal', 'effortless', 'light', 'breezy'],
    'clean girl': ['slicked back', 'natural', 'minimalist', 'gold jewelry', 'simple', 'fresh']
}

# Enhanced formality understanding
FORMALITY_LEVELS = {
    'casual': {
        'keywords': ['casual', 'relaxed', 'comfortable', 'everyday', 'laid-back', 'easy', 'informal'],
        'suitable': ['t-shirt', 'jeans', 'sneakers', 'hoodie', 'shorts', 'casual dress'],
        'score_boost': 0.2
    },
    'smart-casual': {
        'keywords': ['smart casual', 'business casual', 'dressy casual', 'neat', 'polished casual'],
        'suitable': ['polo', 'chinos', 'blouse', 'blazer', 'loafers', 'ankle boots'],
        'score_boost': 0.15
    },
    'business': {
        'keywords': ['business', 'professional', 'office', 'work', 'corporate', 'meeting'],
        'suitable': ['dress shirt', 'slacks', 'blazer', 'suit', 'oxford shoes', 'pumps'],
        'score_boost': 0.15
    },
    'formal': {
        'keywords': ['formal', 'elegant', 'sophisticated', 'dressy', 'gala', 'black tie'],
        'suitable': ['suit', 'dress', 'heels', 'tie', 'gown', 'dress shoes', 'formal wear'],
        'score_boost': 0.2
    }
}

# Enhanced occasion understanding
OCCASIONS = {
    'work': ['work', 'office', 'meeting', 'professional', 'business', 'corporate', 'interview'],
    'casual': ['casual', 'everyday', 'running errands', 'shopping', 'grocery', 'relaxed'],
    'party': ['party', 'night out', 'club', 'bar', 'evening', 'celebration', 'nightclub'],
    'date': ['date', 'romantic', 'dinner date', 'first date', 'anniversary'],
    'gym': ['gym', 'workout', 'exercise', 'fitness', 'sports', 'athletic', 'yoga'],
    'beach': ['beach', 'pool', 'swim', 'vacation', 'resort', 'tropical'],
    'wedding': ['wedding', 'formal event', 'ceremony', 'reception', 'gala'],
    'brunch': ['brunch', 'lunch', 'coffee', 'breakfast', 'cafe', 'tea']
}

# Weather appropriateness
WEATHER_ITEMS = {
    'hot': ['light', 'breathable', 'cotton', 'linen', 'shorts', 'tank', 'sundress', 'sandals'],
    'cold': ['warm', 'coat', 'jacket', 'sweater', 'boots', 'layered', 'wool', 'thermal'],
    'rainy': ['waterproof', 'jacket', 'boots', 'umbrella', 'rain coat', 'water-resistant']
}

# -----------------------------------------------------
# Global styling
# -----------------------------------------------------
def set_global_style():
    bg_style = f"""
        background-image: url("data:image/png;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """ if bg_b64 else "background: linear-gradient(135deg, #1b1412 0%, #2d1f1a 100%);"
    
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
            
            [data-testid="stAppViewContainer"] {{
                {bg_style}
            }}
            
            [data-testid="stAppViewContainer"]::before {{
                content: "";
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background-color: rgba(0,0,0,0.5);
                z-index: -1;
            }}
            
            * {{
                font-family: 'Poppins', sans-serif;
            }}
            
            .stMarkdown, p, div, span, label {{
                color: #f5e0c0 !important;
            }}
            
            h1, h2, h3 {{
                color: #e6ccb2 !important;
                font-weight: 700;
            }}
            
            .stButton > button {{
                background-color: #b08968 !important;
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
                font-weight: 600 !important;
                transition: all 0.3s ease;
            }}
            
            .stButton > button:hover {{
                background-color: #a67856 !important;
                transform: scale(1.03);
            }}
            
            .delete-btn {{
                background-color: #c94c4c !important;
            }}
            
            .delete-btn:hover {{
                background-color: #a83939 !important;
            }}
            
            .card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            
            .match-score {{
                background: linear-gradient(135deg, #b08968 0%, #d7b98e 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                display: inline-block;
                margin: 5px 0;
            }}
            
            .outfit-card {{
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(15px);
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                border: 2px solid rgba(215, 185, 142, 0.3);
            }}
            
            .preference-badge {{
                background: rgba(176, 137, 104, 0.3);
                padding: 5px 12px;
                border-radius: 15px;
                display: inline-block;
                margin: 3px;
                font-size: 14px;
            }}
            
            .section-box {{
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                border: 2px solid rgba(215, 185, 142, 0.2);
            }}
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# Load CLIP model
# -----------------------------------------------------
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# -----------------------------------------------------
# Feature extraction
# -----------------------------------------------------
def extract_image_features(image, model, preprocess, device):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def extract_text_features(text, model, device):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def calculate_similarity(query_features, item_features):
    return np.dot(query_features, item_features.T)[0][0]

# -----------------------------------------------------
# ENHANCED QUERY UNDERSTANDING
# -----------------------------------------------------
def understand_query_advanced(query, preferences):
    """
    Advanced query understanding with conflict resolution
    Handles mixed formality requests like "formal but casual"
    """
    query_lower = query.lower()
    
    # Detect formality levels with scores
    formality_scores = {}
    detected_formality = None
    primary_formality = None
    secondary_formality = None
    
    for level, data in FORMALITY_LEVELS.items():
        score = sum(1 for keyword in data['keywords'] if keyword in query_lower)
        if score > 0:
            formality_scores[level] = score
    
    # Handle mixed formality (e.g., "formal but casual")
    if len(formality_scores) >= 2:
        sorted_formality = sorted(formality_scores.items(), key=lambda x: x[1], reverse=True)
        primary_formality = sorted_formality[0][0]
        secondary_formality = sorted_formality[1][0]
        
        # Check for mixing indicators
        if any(word in query_lower for word in ['but', 'yet', 'though', 'however', 'with', 'mixed']):
            detected_formality = 'smart-casual'  # Compromise between formal and casual
        else:
            detected_formality = primary_formality
    elif formality_scores:
        detected_formality = max(formality_scores, key=formality_scores.get)
    else:
        detected_formality = 'casual'  # Default
    
    # Detect trends
    detected_trends = []
    for trend, keywords in FASHION_TRENDS.items():
        if any(keyword in query_lower for keyword in keywords) or trend in query_lower:
            detected_trends.append(trend)
    
    # Detect occasions
    detected_occasion = None
    for occasion, keywords in OCCASIONS.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_occasion = occasion
            break
    
    # Detect weather preferences (from query or preferences)
    detected_weather = None
    weather_map = {
        'sunny/hot': 'hot',
        'cold': 'cold',
        'rainy': 'rainy',
        'mild': 'mild'
    }
    
    if preferences.get('weather') and preferences['weather'] in weather_map:
        detected_weather = weather_map[preferences['weather']]
    
    for weather, keywords in WEATHER_ITEMS.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_weather = weather
            break
    
    return {
        'formality': detected_formality,
        'primary_formality': primary_formality,
        'secondary_formality': secondary_formality,
        'trends': detected_trends,
        'occasion': detected_occasion,
        'weather': detected_weather,
        'original_query': query,
        'is_mixed_formality': primary_formality and secondary_formality
    }

# -----------------------------------------------------
# IMPROVED OUTFIT GENERATION
# -----------------------------------------------------
def generate_outfit_recommendations_advanced(query_analysis, wardrobe_items, model, device, preferences):
    """
    Enhanced outfit generation with better scoring and preference handling
    """
    if not wardrobe_items:
        return [], []
    
    # Build comprehensive query
    query_parts = [query_analysis['original_query']]
    
    # Add formality
    if query_analysis['formality']:
        query_parts.append(query_analysis['formality'])
        suitable_items = FORMALITY_LEVELS[query_analysis['formality']]['suitable']
        query_parts.extend(suitable_items[:3])
    
    # Add trends
    if query_analysis['trends']:
        query_parts.extend(query_analysis['trends'])
    
    # Add occasion
    if query_analysis['occasion']:
        query_parts.append(query_analysis['occasion'])
    
    # Add weather
    if query_analysis['weather']:
        query_parts.append(query_analysis['weather'])
        weather_keywords = WEATHER_ITEMS.get(query_analysis['weather'], [])
        query_parts.extend(weather_keywords[:3])
    
    # Add user preferences
    if preferences.get('style_preference'):
        query_parts.extend(preferences['style_preference'])
    
    enhanced_query = ' '.join(query_parts)
    query_features = extract_text_features(enhanced_query, model, device)
    
    # Score each item with multiple factors
    scored_items = []
    for item in wardrobe_items:
        # Base similarity
        similarity = calculate_similarity(query_features, item['features'])
        
        # Formality matching bonus
        formality_bonus = 0
        item_formality = item.get('formality', 'casual')
        if item_formality == query_analysis['formality']:
            formality_bonus = 0.25
        elif query_analysis['is_mixed_formality']:
            if item_formality == 'smart-casual':
                formality_bonus = 0.20
            elif item_formality in [query_analysis['primary_formality'], query_analysis['secondary_formality']]:
                formality_bonus = 0.15
        
        # Weather appropriateness
        weather_bonus = 0
        if query_analysis['weather']:
            item_desc = (item.get('description', '') + ' ' + item.get('category', '')).lower()
            weather_keywords = WEATHER_ITEMS.get(query_analysis['weather'], [])
            if any(keyword in item_desc for keyword in weather_keywords):
                weather_bonus = 0.15
            
            if query_analysis['weather'] == 'hot' and 'summer' in item.get('season', []):
                weather_bonus += 0.1
            elif query_analysis['weather'] == 'cold' and 'winter' in item.get('season', []):
                weather_bonus += 0.1
        
        # Color preference bonus
        color_bonus = 0
        if preferences.get('color_preference'):
            if item.get('color', '').lower() in [c.lower() for c in preferences['color_preference']]:
                color_bonus = 0.1
        
        # Calculate final score
        final_score = min(similarity + formality_bonus + weather_bonus + color_bonus, 1.0)
        
        scored_items.append({
            **item,
            'match_score': final_score,
            'formality_bonus': formality_bonus,
            'weather_bonus': weather_bonus
        })
    
    # Sort by score
    scored_items.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Create outfit combinations
    categories = {}
    for item in scored_items:
        category = item.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append(item)
    
    outfits = []
    
    # Generate dress-based outfits
    if 'dress' in categories:
        for dress in categories['dress'][:3]:
            outfit = {'items': [dress], 'avg_score': dress['match_score']}
            if 'shoes' in categories:
                outfit['items'].append(categories['shoes'][0])
            if 'accessories' in categories:
                outfit['items'].append(categories['accessories'][0])
            outfit['avg_score'] = sum(item['match_score'] for item in outfit['items']) / len(outfit['items'])
            outfits.append(outfit)
    
    # Generate top+bottom combinations
    if 'top' in categories and 'bottom' in categories:
        for top in categories['top'][:3]:
            for bottom in categories['bottom'][:3]:
                outfit = {'items': [top, bottom], 'avg_score': (top['match_score'] + bottom['match_score']) / 2}
                
                if 'outerwear' in categories and query_analysis['weather'] in ['cold', 'rainy']:
                    outfit['items'].append(categories['outerwear'][0])
                
                if 'shoes' in categories:
                    outfit['items'].append(categories['shoes'][0])
                
                if 'accessories' in categories:
                    outfit['items'].append(categories['accessories'][0])
                
                outfit['avg_score'] = sum(item['match_score'] for item in outfit['items']) / len(outfit['items'])
                outfits.append(outfit)
    
    outfits.sort(key=lambda x: x['avg_score'], reverse=True)
    
    return outfits[:5], scored_items[:15]

# -----------------------------------------------------
# AI RESPONSE GENERATION
# -----------------------------------------------------
def generate_ai_response(query_analysis, wardrobe_items):
    """Generate contextual AI response"""
    response_parts = []
    
    response_parts.append("I understand what you're looking for! ")
    
    if query_analysis['is_mixed_formality']:
        response_parts.append(f"You want a {query_analysis['primary_formality']} look with {query_analysis['secondary_formality']} elements - ")
        response_parts.append("I'll find you a smart-casual balance that works perfectly! ")
    elif query_analysis['formality']:
        response_parts.append(f"You're going for a {query_analysis['formality']} style. ")
    
    if query_analysis['occasion']:
        response_parts.append(f"Perfect for a {query_analysis['occasion']} occasion. ")
    
    if query_analysis['weather']:
        response_parts.append(f"I'll make sure it's suitable for {query_analysis['weather']} weather. ")
    
    if query_analysis['trends']:
        response_parts.append(f"Incorporating {', '.join(query_analysis['trends'])} vibes! ")
    
    if not wardrobe_items:
        response_parts.append("\n\nYour wardrobe is empty! Upload some items to get personalized recommendations.")
    else:
        response_parts.append(f"\n\nSearching through your {len(wardrobe_items)} items to create the perfect outfits...")
    
    return ''.join(response_parts)

# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------
def home_page():
    set_global_style()
    
    st.markdown("""
    <div style="text-align: center; margin-top: 100px;">
        <div style="display: inline-flex; gap: 10px; margin-bottom: 10px;">
            <h1 style="font-size: 95px; margin: 0; color: #f5e0c0;">SMART</h1>
            <h1 style="font-size: 95px; margin: 0; color: #d7b98e;">CLOSET</h1>
        </div>
        <p style="font-size: 22px; color: #f5e0c0; opacity: 0.9; margin-bottom: 50px;">
            Where Fashion Meets Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("AI STYLE ASSISTANT", use_container_width=True, key="home_assistant", type="primary"):
            st.session_state.current_page = 'assistant'
            st.rerun()
        
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        if st.button("MY COLLECTIONS", use_container_width=True, key="home_wardrobe"):
            st.session_state.current_page = 'wardrobe'
            st.rerun()

# -----------------------------------------------------
# MY COLLECTIONS PAGE
# -----------------------------------------------------
def upload_page():
    set_global_style()
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #f5e0c0;'>MY COLLECTIONS</h1>
        <p style='color: #d7b98e; font-size: 18px;'>Manage your wardrobe items</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation at top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Back to Home", use_container_width=True, key="wardrobe_home"):
            st.session_state.current_page = 'home'
            st.rerun()
    with col3:
        if st.button("AI Style Assistant", type="primary", use_container_width=True, key="wardrobe_assistant"):
            st.session_state.current_page = 'assistant'
            st.rerun()
    
    st.markdown("---")
    
    # Upload section
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("Add New Items")
    
    uploaded_files = st.file_uploader(
        "Choose images of your clothes",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload clear photos of individual clothing items"
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} items selected")
        
        with st.form("item_details"):
            st.write("**Add details for better recommendations:**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                category = st.selectbox(
                    "Category",
                    ['top', 'bottom', 'dress', 'shoes', 'accessories', 'outerwear', 'other']
                )
                color = st.text_input("Primary Color", placeholder="e.g., blue, black, white")
            
            with col_b:
                season = st.multiselect(
                    "Suitable for",
                    ['spring', 'summer', 'fall', 'winter', 'all-season']
                )
                formality = st.select_slider(
                    "Formality Level",
                    options=['casual', 'smart-casual', 'business', 'formal']
                )
            
            description = st.text_area(
                "Description (optional)",
                placeholder="e.g., cotton t-shirt, perfect for summer, comfortable",
                help="This helps the AI understand your item better"
            )
            
            submitted = st.form_submit_button("Add to Collection", type="primary")
            
            if submitted and uploaded_files:
                if not st.session_state.model_loaded:
                    with st.spinner("Loading AI model... (first time only)"):
                        model, preprocess, device = load_clip_model()
                        st.session_state.model = model
                        st.session_state.preprocess = preprocess
                        st.session_state.device = device
                        st.session_state.model_loaded = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}...")
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    features = extract_image_features(
                        image,
                        st.session_state.model,
                        st.session_state.preprocess,
                        st.session_state.device
                    )
                    
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    import random
                    unique_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}_{idx}"
                    
                    item = {
                        'id': unique_id,
                        'image': img_str,
                        'category': category,
                        'color': color,
                        'season': season,
                        'formality': formality,
                        'description': description,
                        'features': features,
                        'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.wardrobe_items.append(item)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"Successfully added {len(uploaded_files)} items!")
                st.balloons()
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display items
    if st.session_state.wardrobe_items:
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Your Collection ({len(st.session_state.wardrobe_items)} items)")
        with col2:
            if st.button("Clear All Items", use_container_width=True, key="clear_all"):
                st.session_state.wardrobe_items = []
                st.success("Collection cleared!")
                st.rerun()
        
        cols = st.columns(4)
        for idx, item in enumerate(st.session_state.wardrobe_items):
            with cols[idx % 4]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                img_data = base64.b64decode(item['image'])
                img = Image.open(io.BytesIO(img_data))
                st.image(img, use_container_width=True)
                
                st.write(f"**{item['category'].title()}**")
                st.write(f"Color: {item['color']}")
                st.write(f"Formality: {item['formality']}")
                
                if st.button(f"Delete", key=f"delete_{item['id']}", use_container_width=True):
                    st.session_state.wardrobe_items = [
                        i for i in st.session_state.wardrobe_items if i['id'] != item['id']
                    ]
                    st.success(f"Deleted {item['category']}!")
                    st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Your collection is empty. Upload items to get started!")

# -----------------------------------------------------
# ASSISTANT PAGE - COMPLETE VERSION
# -----------------------------------------------------
def assistant_page():
    set_global_style()
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #f5e0c0;'>AI STYLE ASSISTANT</h1>
        <p style='color: #d7b98e; font-size: 18px;'>Let AI create perfect outfits for you</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with preferences
    with st.sidebar:
        st.header("Your Preferences")
        
        st.session_state.user_preferences['weather'] = st.selectbox(
            "Weather",
            ['', 'Sunny/Hot', 'Cold', 'Rainy', 'Mild']
        )
        
        st.session_state.user_preferences['event'] = st.selectbox(
            "Event/Occasion",
            ['', 'Casual Outing', 'Work/Office', 'Date Night', 'Party', 
             'Gym', 'Formal Event', 'Beach', 'Brunch', 'Wedding']
        )
        
        st.session_state.user_preferences['style_preference'] = st.multiselect(
            "Style Preferences",
            ['Barbiecore', 'Cottagecore', 'Dark Academia', 'Streetwear', 
             'Minimalist', 'Y2K', 'Coastal Grandmother', 'Clean Girl']
        )
        
        st.session_state.user_preferences['color_preference'] = st.multiselect(
            "Preferred Colors",
            ['Black', 'White', 'Gray', 'Beige', 'Brown', 'Blue', 'Navy', 
             'Red', 'Pink', 'Green', 'Yellow', 'Purple', 'Orange']
        )
        
        st.markdown("---")
        
        # Show active preferences
        if any(st.session_state.user_preferences.values()):
            st.write("**Active Preferences:**")
            if st.session_state.user_preferences['weather']:
                st.markdown(f"<div class='preference-badge'>🌤️ {st.session_state.user_preferences['weather']}</div>", 
                           unsafe_allow_html=True)
            if st.session_state.user_preferences['event']:
                st.markdown(f"<div class='preference-badge'>📅 {st.session_state.user_preferences['event']}</div>", 
                           unsafe_allow_html=True)
            for style in st.session_state.user_preferences['style_preference']:
                st.markdown(f"<div class='preference-badge'>✨ {style}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        if st.button("Back to Wardrobe", use_container_width=True, key="assistant_wardrobe"):
            st.session_state.current_page = 'wardrobe'
            st.rerun()
        
        if st.button("Home", use_container_width=True, key="assistant_home"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    # Main content
    st.markdown("### 💬 What would you like to wear?")
    
    # Example queries
    with st.expander("💡 Example Queries (Click to see examples)"):
        st.write("""
        **Mixed Formality:**
        - "I want to look formal but dress casually"
        - "Professional outfit but comfortable"
        - "Elegant yet relaxed look"
        
        **Specific Occasions:**
        - "Casual summer barbiecore outfit"
        - "Business meeting but make it stylish"
        - "Date night outfit that's not too formal"
        - "Gym outfit for hot weather"
        
        **Weather-based:**
        - "Something warm for cold weather"
        - "Light and breezy for summer"
        - "Rainy day outfit"
        """)
    
    # Query input
    query_input = st.text_input(
        "Describe your outfit:",
        placeholder="e.g., I want to look formal but dress casually for today's event",
        help="Be specific! Mention formality, occasion, weather, or style preferences"
    )
    
    # Generate button
    if st.button("✨ Generate Outfits", type="primary", use_container_width=True, key="generate"):
        if not query_input:
            st.warning("⚠️ Please describe what you're looking for!")
        elif not st.session_state.wardrobe_items:
            st.error("❌ Your wardrobe is empty! Please upload items first.")
            if st.button("Go to Wardrobe →", type="primary"):
                st.session_state.current_page = 'wardrobe'
                st.rerun()
        else:
            with st.spinner("🔮 Creating perfect outfits for you..."):
                # Load model if needed
                if not st.session_state.model_loaded:
                    model, preprocess, device = load_clip_model()
                    st.session_state.model = model
                    st.session_state.preprocess = preprocess
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                
                # Analyze query with preferences
                query_analysis = understand_query_advanced(
                    query_input,
                    st.session_state.user_preferences
                )
                
                # Generate AI response
                ai_response = generate_ai_response(
                    query_analysis,
                    st.session_state.wardrobe_items
                )
                
                # Display AI understanding
                st.markdown("<div class='outfit-card'>", unsafe_allow_html=True)
                st.markdown("### 🤖 AI Understanding")
                st.info(ai_response)
                
                # Show what AI detected
                col1, col2, col3 = st.columns(3)
                with col1:
                    if query_analysis['formality']:
                        st.write(f"**Formality:** {query_analysis['formality'].title()}")
                        if query_analysis['is_mixed_formality']:
                            st.caption(f"Mixing {query_analysis['primary_formality']} & {query_analysis['secondary_formality']}")
                
                with col2:
                    if query_analysis['occasion']:
                        st.write(f"**Occasion:** {query_analysis['occasion'].title()}")
                
                with col3:
                    if query_analysis['weather']:
                        st.write(f"**Weather:** {query_analysis['weather'].title()}")
                
                if query_analysis['trends']:
                    st.write(f"**Style Trends:** {', '.join(query_analysis['trends']).title()}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Generate outfits
                outfits, individual_items = generate_outfit_recommendations_advanced(
                    query_analysis,
                    st.session_state.wardrobe_items,
                    st.session_state.model,
                    st.session_state.device,
                    st.session_state.user_preferences
                )
                
                # Display outfits
                if outfits:
                    st.markdown("---")
                    st.markdown(f"### 👔 Top {len(outfits)} Outfit Recommendations")
                    
                    for outfit_idx, outfit in enumerate(outfits):
                        match_percentage = int(outfit['avg_score'] * 100)
                        
                        with st.expander(
                            f"✨ Outfit #{outfit_idx + 1} - {match_percentage}% Match",
                            expanded=(outfit_idx == 0)
                        ):
                            st.markdown("<div class='outfit-card'>", unsafe_allow_html=True)
                            
                            # Match score badge
                            st.markdown(
                                f"<div class='match-score'>Match Score: {match_percentage}%</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Display items in columns
                            cols = st.columns(len(outfit['items']))
                            for item_idx, item in enumerate(outfit['items']):
                                with cols[item_idx]:
                                    img_data = base64.b64decode(item['image'])
                                    img = Image.open(io.BytesIO(img_data))
                                    st.image(img, use_container_width=True)
                                    
                                    st.markdown(f"**{item['category'].title()}**")
                                    st.write(f"Color: {item['color']}")
                                    st.write(f"Formality: {item['formality']}")
                                    
                                    item_match = int(item['match_score'] * 100)
                                    st.progress(item['match_score'])
                                    st.caption(f"Item Match: {item_match}%")
                                    
                                    # Show why it matched
                                    if item.get('formality_bonus', 0) > 0:
                                        st.caption("✅ Formality match")
                                    if item.get('weather_bonus', 0) > 0:
                                        st.caption("✅ Weather appropriate")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show individual high-matching items
                    st.markdown("---")
                    st.markdown("### 🌟 Individual Item Matches")
                    
                    cols = st.columns(5)
                    for idx, item in enumerate(individual_items[:10]):
                        with cols[idx % 5]:
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            img_data = base64.b64decode(item['image'])
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)
                            
                            item_match = int(item['match_score'] * 100)
                            st.write(f"**{item['category'].title()}**")
                            st.write(f"{item_match}% match")
                            st.progress(item['match_score'])
                            st.markdown("</div>", unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Couldn't create complete outfits. Try adding more items or adjusting your query!")
                    st.info("💡 Tip: Make sure you have tops, bottoms, and shoes in your wardrobe for better outfit combinations.")

# -----------------------------------------------------
# MAIN ROUTER
# -----------------------------------------------------
if st.session_state.current_page == 'home':
    home_page()
elif st.session_state.current_page == 'wardrobe':
    upload_page()
elif st.session_state.current_page == 'assistant':
    assistant_page()
else:
    st.session_state.current_page = 'home'
    st.rerun()