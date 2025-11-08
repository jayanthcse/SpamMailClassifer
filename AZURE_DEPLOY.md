# 🔵 Azure Deployment Guide (Easiest Cloud Option)

## Method 1: Azure Portal (No Command Line - EASIEST!)

### Step 1: Create Azure Account
1. Go to: https://portal.azure.com
2. Sign up (Free tier includes $200 credit for 30 days)
3. No credit card required for free tier

### Step 2: Create Web App
1. In Azure Portal, click **"Create a resource"**
2. Search for **"Web App"** and click it
3. Click **"Create"**

**Fill in details:**
- **Subscription**: Your subscription
- **Resource Group**: Click "Create new" → Name it `spam-classifier-rg`
- **Name**: `spam-classifier-app` (or any unique name)
  - This becomes your URL: `spam-classifier-app.azurewebsites.net`
- **Publish**: `Code`
- **Runtime stack**: `Python 3.11`
- **Operating System**: `Linux`
- **Region**: Choose closest to you (e.g., `East US`)
- **Pricing Plan**: 
  - Click "Explore pricing plans"
  - Select **"Free F1"** (completely free!)
  - Click "Select"

4. Click **"Review + Create"**
5. Click **"Create"**
6. Wait 1-2 minutes for deployment

### Step 3: Configure Startup Command
1. Go to your Web App in Azure Portal
2. In left menu, find **"Configuration"**
3. Click **"General settings"** tab
4. In **"Startup Command"** box, paste:
   ```
   gunicorn --bind=0.0.0.0 --timeout 600 app:app
   ```
5. Click **"Save"** at top
6. Click **"Continue"** when prompted

### Step 4: Deploy Your Code

**Option A: Using VS Code (EASIEST)**

1. Install VS Code extension: "Azure App Service"
2. In VS Code, click Azure icon in sidebar
3. Sign in to Azure
4. Right-click your app → "Deploy to Web App"
5. Select your project folder
6. Wait 2-3 minutes
7. Done! Visit your URL

**Option B: Using Git (Simple)**

1. In Azure Portal, go to your Web App
2. In left menu, click **"Deployment Center"**
3. Select **"Local Git"**
4. Click **"Save"**
5. Copy the Git URL shown

In your terminal:
```bash
# Add Azure as remote
git remote add azure <paste-the-git-url-here>

# Deploy
git add .
git commit -m "Deploy to Azure"
git push azure main
```

6. Enter Azure credentials when prompted
7. Wait 3-5 minutes for build
8. Done!

**Option C: Using GitHub (Automatic)**

1. Push your code to GitHub first
2. In Azure Portal → Deployment Center
3. Select **"GitHub"**
4. Authorize Azure to access GitHub
5. Select your repository
6. Click **"Save"**
7. Azure will auto-deploy on every push!

### Step 5: Verify Deployment

1. In Azure Portal, go to your Web App
2. Click **"Browse"** at the top
3. Your app should open in browser!
4. URL: `https://your-app-name.azurewebsites.net`

### Troubleshooting

**If app doesn't load:**

1. Check logs:
   - Go to **"Log stream"** in left menu
   - Watch for errors

2. Common issues:
   - **"Application Error"**: Startup command might be wrong
   - **"502 Bad Gateway"**: App is still starting (wait 2 min)
   - **"Module not found"**: Requirements not installed (redeploy)

3. Verify files uploaded:
   - Go to **"SSH"** under Development Tools
   - Click **"Go"**
   - Run: `ls -la` to see files
   - Check if `.joblib` files are there

### Important Notes

**File Size Warning:**
- Your model files are ~6MB total
- Free tier has 1GB storage (plenty of space)
- If you get size errors, you're fine - it's within limits

**Cold Start:**
- Free tier apps "sleep" after 20 min of inactivity
- First request after sleep takes 10-20 seconds
- Subsequent requests are fast
- Upgrade to Basic tier ($13/month) to avoid this

**Environment Variables:**
If you need to add any:
1. Go to **"Configuration"**
2. Click **"Application settings"**
3. Click **"New application setting"**
4. Add key-value pairs

---

## Cost Breakdown

- **Free Tier (F1)**: $0/month
  - 1 GB storage
  - 60 CPU minutes/day
  - Apps sleep after 20 min idle
  - Perfect for testing!

- **Basic Tier (B1)**: ~$13/month
  - 10 GB storage
  - Always on (no sleep)
  - Custom domains
  - Better for production

---

## Quick Commands Reference

```bash
# Check if app is running
curl https://your-app-name.azurewebsites.net/api/health

# View logs
az webapp log tail --name your-app-name --resource-group spam-classifier-rg

# Restart app
az webapp restart --name your-app-name --resource-group spam-classifier-rg
```

---

## Success Checklist

✅ Web App created in Azure  
✅ Startup command configured  
✅ Code deployed (Git/VS Code/GitHub)  
✅ App loads in browser  
✅ Can classify emails  
✅ API endpoint works  

**Your app is live!** 🎉
