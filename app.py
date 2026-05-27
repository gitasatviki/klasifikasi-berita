import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import os
import gdown

st.set_page_config(
    page_title="Klasifikasi Berita",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#load css
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path, 'r') as f:
    css_content = f.read()

st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

#konfigurasi model
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import BERTLSTMClassifier   

BERT_MODEL = 'indobenchmark/indobert-base-p1'  
MAX_LENGTH = 64                                  
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_HIDDEN_DIM = 128   
BEST_NUM_LAYERS = 2     
BEST_DROPOUT    = 0.5   

CATEGORIES = ['ekonomi', 'kesehatan', 'olahraga', 'politik', 'teknologi']


_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_modeltp.pth')
if not os.path.exists(_MODEL_PATH):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    _gdrive_url = "https://drive.google.com/uc?id=1UjANllxhosnF7SZLFbU_tRg5vX3TLGQY"
    gdown.download(_gdrive_url, _MODEL_PATH, quiet=False)

@st.cache_resource(show_spinner="Memuat model IndoBERT + LSTM...")
def load_model():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    model = BERTLSTMClassifier(
        bert_model_name = BERT_MODEL,
        num_classes     = len(CATEGORIES),
        hidden_dim      = BEST_HIDDEN_DIM,
        num_layers      = BEST_NUM_LAYERS,
        dropout         = BEST_DROPOUT,
        freeze_bert     = True,
    ).to(DEVICE)

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_modeltp.pth')
    if not os.path.exists(model_path):
        st.warning("File model tidak ditemukan — berjalan dalam mode DEMO (hasil random).")
        return None, tokenizer

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, tokenizer

_model, _tokenizer = load_model()

def predict_text(text):
    if _model is None:
        import random
        scores = [random.uniform(0.05, 0.95) for _ in CATEGORIES]
        total  = sum(scores)
        scores = [s / total for s in scores]
        idx    = scores.index(max(scores))
        return CATEGORIES[idx], scores[idx], {CATEGORIES[i]: round(scores[i]*100,2) for i in range(len(CATEGORIES))}

    encoding = _tokenizer(
        text,
        add_special_tokens    = True,
        max_length            = MAX_LENGTH,
        padding               = 'max_length',
        truncation            = True,
        return_attention_mask = True,
        return_tensors        = 'pt'
    )
    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = _model(input_ids, attention_mask)
        probs  = F.softmax(logits, dim=1)[0]

    idx       = int(torch.argmax(probs).item())
    conf_dict = {CATEGORIES[i]: round(float(probs[i])*100, 2) for i in range(len(CATEGORIES))}
    return CATEGORIES[idx], float(probs[idx]), conf_dict


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def scrape_kompas(keyword, max_articles=10):
    articles = []
    per_page = 10
    page = 1

    try:
        while len(articles) < max_articles:
            url = f"https://search.kompas.com/search/?q={keyword}&orderby=latest&per_page={per_page}&page={page}"
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.content, "html.parser")

            items = soup.find_all('div', class_='articleItem')
            if not items:
                items = soup.find_all('div', class_='content-asf')
            if not items:
                break  

            for item in items:
                if len(articles) >= max_articles:
                    break

                link_tag = item.find('a', class_='article-link') or item.find('a')
                title_tag = item.find('h2', class_='articleTitle') or item.find('h3') or item.find('h2')
                category_tag = item.find('div', class_='articlePost-subtitle')
                date_tag = item.find('div', class_='articlePost-date')

                if not (link_tag and title_tag):
                    continue

                link = link_tag.get('href', '')
                title = title_tag.get_text(strip=True)
                category = category_tag.get_text(strip=True) if category_tag else ''
                date = date_tag.get_text(strip=True) if date_tag else ''

                snippet = ''
                try:
                    art_r = requests.get(link, headers=HEADERS, timeout=8)
                    if art_r.status_code == 200:
                        art_soup = BeautifulSoup(art_r.content, "html.parser")
                        content_div = art_soup.find('div', class_='read__content')
                        if content_div:
                            paras = content_div.find_all('p')
                            clean = []
                            for p in paras:
                                t = p.get_text(strip=True)
                                if t and not t.startswith('Baca juga') and not t.startswith('DOK.') and not t.startswith('Dok.'):
                                    clean.append(t)
                            snippet = ' '.join(clean)
                except Exception:
                    pass

                if len(snippet.strip()) > 30:
                    articles.append({
                        'title': title,
                        'url': link,
                        'source': 'Kompas',
                        'category_label': category,
                        'date': date,
                        'snippet': snippet
                    })
                time.sleep(0.3)

            if len(items) < per_page:
                break

            page += 1
            time.sleep(0.5)

    except Exception as e:
        st.warning(f"Gagal scraping Kompas: {e}")
    return articles


def scrape_detik(keyword, max_articles=10):
    articles = []
    page = 1

    try:
        while len(articles) < max_articles:
            url = f"https://www.detik.com/search/searchall?query={keyword}&sortby=time&page={page}"
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.content, "html.parser")

            items = soup.find_all('article')
            if not items:
                break  

            for item in items:
                if len(articles) >= max_articles:
                    break

                link_tag = item.find('a')
                if not link_tag:
                    continue
                link = link_tag.get('href', '')
                if not link.startswith('http'):
                    link = 'https:' + link

                title = ''
                date = ''
                snippet = ''
                try:
                    art_r = requests.get(link, headers=HEADERS, timeout=8)
                    if art_r.status_code == 200:
                        art_soup = BeautifulSoup(art_r.content, "html.parser")

                        meta_title = art_soup.find('meta', property='og:title')
                        title = meta_title['content'] if meta_title else link_tag.get_text(strip=True)

                        meta_date = art_soup.find('meta', attrs={'name': 'publishdate'})
                        date = meta_date['content'][:10] if meta_date else ''

                        content_div = art_soup.find('div', class_='detail__body-text itp_bodycontent')
                        if content_div:
                            paras = content_div.find_all('p')
                            clean = [p.get_text(strip=True) for p in paras
                                     if p.get_text(strip=True) and not p.get_text(strip=True).startswith('Baca juga')]
                            snippet = ' '.join(clean)
                except Exception:
                    title = link_tag.get_text(strip=True) or 'Artikel Detik'

                if not title:
                    continue

                if len(snippet.strip()) > 30:
                    articles.append({
                        'title': title,
                        'url': link,
                        'source': 'Detik',
                        'category_label': '',
                        'date': date,
                        'snippet': snippet
                    })
                time.sleep(0.3)

            if len(items) < 10:
                break  

            page += 1
            time.sleep(0.5)

    except Exception as e:
        st.warning(f"Gagal scraping Detik: {e}")
    return articles

def render_single_result(category, confidence, all_scores, meta=""):
    bars = ""
    for cat_name, cat_score in sorted(all_scores.items(), key=lambda x: -x[1]):
        bar_width = str(cat_score) + "%"
        bars += (
            '<div class="score-row">'
            '<span class="score-name">' + cat_name + '</span>'
            '<div class="score-track"><div class="score-bar" style="width:' + bar_width + '"></div></div>'
            '<span class="score-pct">' + str(cat_score) + '%</span>'
            '</div>'
        )

    conf_pct     = str(round(confidence * 100, 2))
    meta_html    = (" &nbsp;&middot;&nbsp; " + meta) if meta else ""
    bar_fill_w   = conf_pct + "%"

    html = (
        '<div class="result-card">'
        '<div class="result-header">Hasil Klasifikasi</div>'
        '<div class="result-body">'
        '<div class="result-badge">' + category + '</div>'
        '<p class="result-meta">Kepercayaan: <strong>' + conf_pct + '%</strong>' + meta_html + '</p>'
        '<div class="bar-label"><span>Tingkat Kepercayaan</span><span>' + conf_pct + '%</span></div>'
        '<div class="bar-track"><div class="bar-fill" style="width:' + bar_fill_w + '"></div></div>'
        '<div class="all-scores">'
        '<p style="font-size:14px;font-weight:600;margin-bottom:12px">Distribusi Skor Semua Kategori</p>'
        + bars +
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_article_list(articles):
    valid_articles = [a for a in articles if len((a.get('snippet') or '').strip()) > 30]
    skipped = len(articles) - len(valid_articles)

    st.markdown(
        '<div style="font-size:13px;color:#6B7280;margin-bottom:16px">'
        '<strong>' + str(len(valid_articles)) + ' artikel ditemukan</strong></div>',
        unsafe_allow_html=True
    )

    if not valid_articles:
        st.warning("Tidak ada artikel dengan isi yang berhasil diambil. Coba kata kunci lain.")
        return

    for i, art in enumerate(valid_articles):
        snippet = art.get('snippet') or ''
        has_snippet = True  
        snippet_preview = snippet[:220] + ('...' if len(snippet) > 220 else '')

        meta_parts = []
        if art.get('category_label'):
            meta_parts.append(art['category_label'])
        if art.get('date'):
            meta_parts.append(art['date'])
        meta_str = ' &nbsp;•&nbsp; '.join(meta_parts) if meta_parts else ''

        st.markdown(
            '<div class="search-item">'
            '<div class="search-item-title" style="font-size:15px;font-weight:700;margin-bottom:6px">'
            + art['title'] +
            '</div>'
            '<div class="search-item-source">' + art['source']
            + (' &nbsp;•&nbsp; ' + meta_str if meta_str else '') +
            '</div>'
            '<div class="search-item-snippet">' + snippet_preview + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        col_btn, col_link = st.columns([1, 4])
        with col_btn:
            if st.button("Klasifikasi", key=f"classify_art_{i}"):
                if snippet and len(snippet) > 10:
                    with st.spinner("Mengklasifikasi artikel..."):
                        cat, conf, scores = predict_text(snippet)
                    render_single_result(cat, conf, scores, art['source'])
        with col_link:
            if art.get('url', '').startswith('http'):
                st.markdown(
                    '<div style="padding-top:8px">'
                    '<a href="' + art['url'] + '" target="_blank" class="search-item-link" style="font-size:13px">Baca artikel lengkap →</a>'
                    '</div>',
                    unsafe_allow_html=True
                )


def navbar(active_page):
    links = {
        "Home": "home",
        "Klasifikasi Berita": "classify"
    }

    nav_links_html = '<div class="navbar-links">'
    for label, key in links.items():
        active_cls = "active" if key == active_page else ""
        nav_links_html += '<a class="' + active_cls + '" href="?page=' + key + '" target="_self">' + label + '</a>'
    nav_links_html += '</div>'

    st.markdown('<div class="navbar-outer">', unsafe_allow_html=True)

    col_logo, col_links = st.columns([1, 4])

    with col_logo:
        logo_path = os.path.join(os.path.dirname(__file__), 'images', 'logo.png')
        if os.path.exists(logo_path):
            st.image(logo_path, width=140)
        else:
            st.markdown(
                '<div style="font-family:Sora,sans-serif;font-weight:800;font-size:14px;'
                'color:#2563EB;border:2px solid #2563EB;padding:6px 12px;border-radius:8px;'
                'display:inline-block;margin-top:12px;">Klasifikasi Berita</div>',
                unsafe_allow_html=True
            )

    with col_links:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:flex-end;height:64px;">'
            + nav_links_html +
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

#routing
query_params = st.query_params
page = query_params.get("page", "home")

#home
if page == "home":
    navbar("home")

    st.markdown("""
    <div class="hero">
        <div style="flex:1">
            <div class="hero-title">KLASIFIKASI<br>BERITA</div>
            <p class="hero-desc">Sistem klasifikasi berita otomatis dengan menggunakan BERT embedding dan LSTM.</p>
            <a href="?page=classify" target="_self" class="hero-btn">Mulai →</a>
        </div>
        <div class="hero-card">
            <h3>Jenis Berita</h3>
            <div class="cat-item"><div class="cat-dot"></div>Ekonomi</div>
            <div class="cat-item"><div class="cat-dot"></div>Kesehatan</div>
            <div class="cat-item"><div class="cat-dot"></div>Olahraga</div>
            <div class="cat-item"><div class="cat-dot"></div>Teknologi</div>
            <div class="cat-item"><div class="cat-dot"></div>Politik</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="features-section">', unsafe_allow_html=True)
    st.markdown('<div class="features-title">Fitur Utama</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="feat-card">
            <div class="feat-icon" style="background:#DBEAFE"></div>
            <h3>Input Teks</h3>
            <p>Salin dan tempel teks berita langsung ke sistem untuk diklasifikasikan.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feat-card">
            <div class="feat-icon" style="background:#DCFCE7"></div>
            <h3>Upload File</h3>
            <p>Unggah file .txt berisi artikel berita untuk diklasifikasi.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feat-card">
            <div class="feat-icon" style="background:#FEF3C7"></div>
            <h3>Scraping Online</h3>
            <p>Cari dan klasifikasi berita langsung dari Kompas dan Detik.</p>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#klasifikasi
elif page == "classify":
    navbar("classify")

    st.markdown("""
    <div class="page-hero">
        <div class="page-title">Klasifikasi Berita</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="classify-wrapper">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Input Text", "Upload File", "Search Online"])

    #input text
    with tab1:
        st.markdown('<label style="font-size:14px;font-weight:600">Masukkan Berita</label>', unsafe_allow_html=True)
        text_input = st.text_area(
            label="",
            placeholder="Tempel atau ketik teks berita di sini... (minimal 20 karakter)",
            height=200,
            key="text_input",
            label_visibility="collapsed"
        )
        if text_input:
            word_count = len(text_input.split())
            st.markdown(f'<div style="font-size:12px;color:#6B7280;text-align:right;margin-top:-12px">{word_count} kata · {len(text_input)} karakter</div>', unsafe_allow_html=True)

        if st.button("Analyze", key="btn_text"):
            if not text_input or len(text_input.strip()) < 20:
                st.error("Teks terlalu pendek. Minimal 20 karakter.")
            else:
                with st.spinner("Mengklasifikasi teks..."):
                    category, confidence, all_scores = predict_text(text_input)
                meta = f"{len(text_input.split())} kata · {len(text_input)} karakter"
                render_single_result(category, confidence, all_scores, meta)

    #upload file
    with tab2:
        st.markdown('<label style="font-size:14px;font-weight:600">Upload File Berita (.txt)</label>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="",
            type=["txt"],
            label_visibility="collapsed",
            key="file_upload"
        )
        if uploaded:
            st.markdown(f'<div style="font-size:13px;color:#6B7280;padding:10px 14px;background:#F3F4F6;border-radius:8px;border-left:3px solid #2563EB;margin-bottom:8px"><strong>{uploaded.name}</strong> &nbsp;·&nbsp; {round(uploaded.size/1024,1)} KB</div>', unsafe_allow_html=True)

        if st.button("Analyze File", key="btn_file"):
            if not uploaded:
                st.error("Pilih file terlebih dahulu.")
            else:
                content = uploaded.read().decode("utf-8", errors="ignore").strip()
                if not content:
                    st.error("File kosong.")
                else:
                    with st.spinner("Membaca dan menganalisis file..."):
                        category, confidence, all_scores = predict_text(content)
                    meta = f"{uploaded.name} · {len(content.split())} kata"
                    render_single_result(category, confidence, all_scores, meta)

    #scraping
    with tab3:
        st.markdown('<label style="font-size:14px;font-weight:600">Kata Kunci Berita</label>', unsafe_allow_html=True)
        keyword = st.text_input(
            label="",
            placeholder="Contoh: pemilu, jantung, dll...",
            key="search_keyword",
            label_visibility="collapsed"
        )
        st.markdown('<label style="font-size:14px;font-weight:600">Sumber Berita</label>', unsafe_allow_html=True)
        source = st.radio(
            label="",
            options=["Semua", "Kompas", "Detik"],
            horizontal=True,
            key="source_radio",
            label_visibility="collapsed"
        )

        if st.button("Cari Berita", key="btn_search"):
            if not keyword:
                st.error("Masukkan kata kunci terlebih dahulu.")
            else:
                with st.spinner(f"Mencari berita tentang '{keyword}' dari {source}..."):
                    articles = []
                    if source == "Semua":
                        articles += scrape_kompas(keyword, 15)
                        articles += scrape_detik(keyword, 15)
                    elif source == "Kompas":
                        articles += scrape_kompas(keyword, 30)
                    elif source == "Detik":
                        articles += scrape_detik(keyword, 30)

                if not articles:
                    st.warning("Tidak ada artikel ditemukan. Coba kata kunci lain.")
                else:
                    st.session_state['search_articles'] = articles

        if 'search_articles' in st.session_state and st.session_state['search_articles']:
            st.markdown("<hr style='border:1px solid #E5E7EB;margin:20px 0'>", unsafe_allow_html=True)
            render_article_list(st.session_state['search_articles'])

    st.markdown('</div>', unsafe_allow_html=True)