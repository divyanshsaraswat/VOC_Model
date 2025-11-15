# 🚀 Deployment Guide

## Deployment Instructions for All Platforms

### 📋 Prerequisites

1. ✅ `requirements.txt` file exists
2. ✅ `app.py` file exists
3. ✅ `model.pkl` file ready
4. ✅ All files committed to GitHub repository

---

## 🌐 Streamlit Cloud (Easiest)

### Steps:

1. **Create GitHub Repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/voc-streamlit.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in and connect GitHub account
   - Click "New app" button
   - Select repository: `yourusername/voc-streamlit`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Automatic Detection:**
   - Streamlit Cloud automatically detects `requirements.txt`
   - Dependencies are automatically installed
   - App is deployed!

**✅ Done!** Your app will be live at `https://your-app-name.streamlit.app`

---

## 🤗 HuggingFace Spaces

### Steps:

1. **Create HuggingFace Account:**
   - Sign up at [https://huggingface.co](https://huggingface.co)

2. **Create New Space:**
   - Profile → Spaces → "Create new Space"
   - Space name: `voc-streamlit` (or any other)
   - SDK: Select **Streamlit**
   - Visibility: Public/Private
   - Click "Create Space"

3. **Upload Files:**
   - Method 1: Git push (Recommended)
     ```bash
     git clone https://huggingface.co/spaces/yourusername/voc-streamlit
     cd voc-streamlit
     # Copy all your files here
     git add .
     git commit -m "Add Streamlit app"
     git push
     ```
   
   - Method 2: Direct upload via Web UI
     - Upload `app.py`, `requirements.txt`, `model.pkl` files

4. **Deploy:**
   - HuggingFace automatically builds and deploys the app
   - Check "Logs" tab to view logs

**✅ Done!** Your app will be live at `https://huggingface.co/spaces/yourusername/voc-streamlit`

---

## 🚂 Railway

### Steps:

1. **Create Railway Account:**
   - Sign up at [https://railway.app](https://railway.app)
   - Connect GitHub account

2. **Create New Project:**
   - "New Project" → "Deploy from GitHub repo"
   - Select repository: `yourusername/voc-streamlit`

3. **Configure:**
   - Service type: **Web Service** (automatic detect)
   - Build command: (leave empty - auto detect)
   - Start command: 
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - Environment: Python 3.9 or higher

4. **Deploy:**
   - Railway automatically detects and deploys
   - Check logs

**✅ Done!** Your app will be live at `https://your-app-name.up.railway.app`

---

## 🎨 Render

### Steps:

1. **Create Render Account:**
   - Sign up at [https://render.com](https://render.com)
   - Connect GitHub account

2. **Create New Web Service:**
   - Dashboard → "New" → "Web Service"
   - Connect repository: `yourusername/voc-streamlit`

3. **Configure:**
   - Name: `voc-streamlit` (or any other)
   - Environment: **Python 3**
   - Build Command: (leave empty - auto detect)
   - Start Command:
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - Plan: Free/Paid (Free tier available)

4. **Deploy:**
   - Click "Create Web Service"
   - Render automatically builds and deploys
   - Check build logs

**✅ Done!** Your app will be live at `https://voc-streamlit.onrender.com`

---

## 🐳 Docker (Advanced)

### Steps:

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8501

   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run:**
   ```bash
   docker build -t voc-streamlit .
   docker run -p 8501:8501 voc-streamlit
   ```

3. **Deploy on Cloud Platforms:**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform

---

## ⚙️ Environment Variables (Optional)

If you need environment variables:

1. **Streamlit Cloud:**
   - App settings → "Secrets" → Add secrets

2. **HuggingFace Spaces:**
   - Settings → "Repository secrets" → Add secrets

3. **Railway/Render:**
   - Environment tab → Add variables

---

## 🔧 Troubleshooting

### Common Issues:

1. **Model file not found:**
   - Ensure `model.pkl` is in repository
   - Check file path is correct

2. **Dependencies installation failed:**
   - Check `requirements.txt` syntax
   - Verify package versions are compatible

3. **Port binding errors:**
   - Ensure using `$PORT` environment variable
   - Use `--server.address 0.0.0.0` flag

4. **Memory issues:**
   - Large models (>500MB) may require paid tiers
   - Consider model optimization

---

## 📊 Comparison

| Platform | Free Tier | Build Time | Auto Deploy | Easiest |
|----------|-----------|------------|-------------|---------|
| Streamlit Cloud | ✅ | Fast | ✅ | ⭐⭐⭐⭐⭐ |
| HuggingFace Spaces | ✅ | Medium | ✅ | ⭐⭐⭐⭐ |
| Railway | ✅ (Limited) | Fast | ✅ | ⭐⭐⭐⭐ |
| Render | ✅ | Medium | ✅ | ⭐⭐⭐ |

**Recommendation:** Streamlit Cloud is easiest for beginners

---

## ✅ Post-Deployment Checklist

- [ ] App successfully loads
- [ ] Model loads without errors
- [ ] Input fields work correctly
- [ ] Predictions run successfully
- [ ] Error handling works
- [ ] Mobile responsive (check on phone)
- [ ] Share link with others for testing

---

**🎉 Congratulations!** Your ML app is now live and ready to use!
