# Deployment Guide: Predictive Maintenance Explainability Lab

This project is fully prepared for containerized and cloud deployment. Choose any of the deployment options below:

---

## Option 1: Streamlit Community Cloud (Recommended, 100% Free)

Streamlit Community Cloud is the fastest, zero-cost way to deploy Streamlit applications with an instant public `.streamlit.app` URL.

### Step 1: Push Code to GitHub
1. Create a new repository on [GitHub](https://github.com/new) named `predictive-maintenance-lab` (leave it empty, without adding a README or .gitignore).
2. In your local terminal, run:
```bash
cd "c:\Users\Aayush Anand\Desktop\Mini-Proj"
git remote add origin https://github.com/<your-github-username>/predictive-maintenance-lab.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
2. Click **New app**.
3. Fill in the deployment details:
   - **Repository:** `<your-github-username>/predictive-maintenance-lab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**
5. Your app will build and go live within 60 seconds with a permanent public link:
   `https://<your-chosen-subdomain>.streamlit.app`

---

## Option 2: Docker Container (Local / AWS / DigitalOcean / Render)

The project includes a production-ready `Dockerfile` and `.dockerignore`.

### Build and Run Locally:
```bash
# Build the Docker image
docker build -t predictive-maintenance-lab .

# Run container on port 8501
docker run -d -p 8501:8501 --name pdm-lab predictive-maintenance-lab
```
Access at: `http://localhost:8501`

### Deploy on Render / Railway:
1. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
2. Choose **Web Service**.
3. Select Environment: **Docker** (it will automatically detect the `Dockerfile`).
4. Port: `8501`.
5. Click **Create Web Service**.

---

## Option 3: Hugging Face Spaces (Free)

1. Create a free account at [Hugging Face](https://huggingface.co).
2. Go to **Spaces** &rarr; **Create new Space**.
3. Set:
   - Space name: `predictive-maintenance-lab`
   - License: `MIT`
   - Space SDK: **Streamlit**
4. Push your repository to the Hugging Face Git remote:
```bash
git remote add hf https://huggingface.co/spaces/<your-hf-username>/predictive-maintenance-lab
git push hf main
```
Your Space will build and be immediately interactive and publicly shareable.
