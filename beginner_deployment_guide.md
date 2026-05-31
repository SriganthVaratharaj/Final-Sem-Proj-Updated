# Beginner-Friendly Deployment Guide 🚀

Hi! Welcome to the guide. If you have never hosted a project before, don't worry! This guide is designed to explain everything in simple, step-by-step terms so that you can deploy your application successfully.

Here is what we are deploying:
1. **Database**: MongoDB Atlas (Cloud Database - Free)
2. **Backend**: Hugging Face Spaces (Python Web Service - Free)
3. **Frontend**: Vercel (Static Web App Hosting - Free)
4. **AI model**: Kaggle Notebook (Runs the deep learning code - Free)

---

## 🛠️ Step 1: Set up MongoDB Atlas (Cloud Database)
We need to move our database from your local machine to the cloud so that it is always online.

1. **Sign Up**: Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) and create a free account.
2. **Create Database**: Click **Deploy a Database**.
3. **Choose Free Tier**: Choose **M0 (Free)**. Select any region (like AWS us-east-1). Click **Create**.
4. **Security Settings**:
   * Create a database user (e.g. username: `admin`, password: generate a strong password and save it somewhere!).
   * Under **IP Access List**, add `0.0.0.0/0` (Allow access from anywhere). This is necessary because Vercel/Hugging Face servers change their location dynamically.
5. **Get Connection String**:
   * Go to your Database dashboard, click **Connect**.
   * Select **Drivers** (Python).
   * Copy the connection string. It will look like this:
     `mongodb+srv://admin:<password>@cluster0.xxxx.mongodb.net/?retryWrites=true&w=majority`
   * Replace `<password>` with your database user password. Save this URL! We will need it for the backend.

---

## 🐍 Step 2: Deploy Backend to Hugging Face Spaces
Hugging Face will host your FastAPI code. Since it supports Docker, it will build your environment and run it 24/7.

1. **Sign Up**: Create an account on [Hugging Face](https://huggingface.co/).
2. **Create Space**:
   * Click on your Profile (top right) > **New Space**.
   * Give it a name (e.g., `invoice-ai-backend`).
   * **License**: Open source (e.g., `mit`).
   * **Space SDK**: Select **Docker**.
   * **Docker Template**: Select **Blank**.
   * **Space Hardware**: CPU Basic (Free).
   * **Visibility**: Public (so the frontend can call it).
   * Click **Create Space**.
3. **Upload Code**:
   * Once created, click on the **Files** tab in your Space.
   * Click **Add file** > **Upload files**.
   * Upload all files inside your **`backend/`** directory (including `main.py`, `config.py`, the new `db/` folder, `requirements.txt`, and the `Dockerfile` we created).
   * Commit the files. Hugging Face will automatically start building your Docker container!
4. **Add Environment Variables**:
   * Go to **Settings** tab in your Hugging Face Space.
   * Scroll down to **Variables and Secrets**.
   * Click **New Secret** and add:
     * Name: `MONGO_URI`
     * Value: Your connection string from Step 1 (e.g., `mongodb+srv://admin:my-password@cluster0...`).
     * Click Save.
   * Add another secret:
     * Name: `MONGO_DB_NAME`
     * Value: `invoice_ai`
5. **Find your Live Backend URL**:
   * Go to the top of your Space page.
   * Click the three dots `...` (top right) > **Embed this Space**.
   * Copy the **Direct URL** (e.g., `https://username-space-name.hf.space`). This is your backend's permanent API URL!

---

## 🎨 Step 3: Deploy Frontend to Vercel
Vercel will compile your Vite frontend and host it on a very fast CDN.

1. **Push Frontend to GitHub**: Ensure your latest changes are pushed to your GitHub repository.
2. **Sign Up**: Go to [Vercel](https://vercel.com/) and sign up using your GitHub account.
3. **Import Project**:
   * Click **Add New** > **Project**.
   * Select your GitHub repository (`Final-Sem-Proj-Updated`).
4. **Configure Project**:
   * **Root Directory**: Click Edit and select the **`frontend`** directory.
   * **Build & Development Settings**: Keep Vercel's default settings (Vite compiles automatically).
   * **Environment Variables**:
     * Add Name: `VITE_API_URL`
     * Value: Your live Hugging Face URL from Step 2 (e.g., `https://username-space-name.hf.space`).
5. **Deploy**: Click **Deploy**. Vercel will build and give you a free live URL (e.g., `my-project.vercel.app`)!

---

## 🤖 Step 4: Run AI Model on Kaggle (No Paste Needed!)
Now, we connect your Kaggle AI model to the backend.

1. Open your Kaggle Notebook.
2. Install `cloudflared` in Kaggle to create the remote tunnel:
   ```bash
   !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !dpkg -i cloudflared-linux-amd64.deb
   ```
3. Run the tunnel in the background:
   ```bash
   !nohup cloudflared tunnel --url http://localhost:8000 > cloudflared.log 2>&1 &
   ```
4. Put this script in your notebook to **automatically sync** the VLM URL to the backend. Replace the `BACKEND_BASE_URL` with your Hugging Face space URL:
   ```python
   import time
   import re
   import urllib.request
   import json

   time.sleep(5) # Wait for tunnel to start

   # YOUR HUGGING FACE BACKEND DIRECT URL
   BACKEND_BASE_URL = "https://username-space-name.hf.space" 

   try:
       with open("cloudflared.log", "r") as f:
           log_content = f.read()
       
       # Extract the cloudflare URL from logs
       match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', log_content)
       if match:
           vlm_url = match.group(0)
           print("Generated Kaggle VLM URL:", vlm_url)
           
           # Send POST request to backend to update VLM URL
           data = json.dumps({"vlm_url": vlm_url}).encode("utf-8")
           req = urllib.request.Request(
               f"{BACKEND_BASE_URL}/api/settings/vlm_url",
               data=data,
               headers={"Content-Type": "application/json"},
               method="POST"
           )
           with urllib.request.urlopen(req) as res:
               print("🎉 Successfully synced URL to Hugging Face:", json.loads(res.read().decode()))
       else:
           print("❌ Cloudflare URL not found in logs yet. Run this cell again in a few seconds.")
   except Exception as e:
       print("❌ Sync failed:", e)
   ```

---

## 🎉 How it all connects!
1. When you run your **Kaggle Notebook**, it starts and generates a temporary Cloudflare link.
2. The notebook immediately sends this link to your **Hugging Face backend**.
3. The backend saves this link in **MongoDB Atlas** so it remembers it.
4. When a user visits your **Vercel frontend**, they upload an invoice.
5. The frontend calls the backend on **Hugging Face**.
6. The backend reads the active Kaggle link from **MongoDB Atlas** and queries the model!
7. The results are processed, saved in **MongoDB Atlas**, and displayed to the user on the screen.

**You are done! 100% cloud-hosted and 100% free!**
