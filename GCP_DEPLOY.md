# 🔴 GCP Deployment Guide (Google Cloud Platform)

## Prerequisites
- Google Cloud account (free tier: $300 credit for 90 days)
- gcloud CLI installed (optional)

## Method: Google App Engine (Easiest GCP Option)

### Step 1: Create app.yaml

Create this file in your project root:

```yaml
runtime: python311

entrypoint: gunicorn -b :$PORT app:app

instance_class: F1

automatic_scaling:
  max_instances: 1
```

### Step 2: Deploy via Console

1. **Go to**: https://console.cloud.google.com
2. **Create new project**: "spam-classifier"
3. **Enable App Engine**:
   - Search for "App Engine" in search bar
   - Click "Create Application"
   - Choose region (us-central1)
4. **Open Cloud Shell** (icon at top right)
5. **Upload your code**:
   - Click "Upload" button in Cloud Shell
   - Select all your project files
6. **Deploy**:
   ```bash
   gcloud app deploy
   ```
7. **Wait**: 3-5 minutes
8. **Access**:
   ```bash
   gcloud app browse
   ```

### Step 3: View Your App

- URL will be: `https://spam-classifier.uc.r.appspot.com`

### Troubleshooting

**View logs:**
```bash
gcloud app logs tail -s default
```

Or in Console:
1. Go to "Logging" in left menu
2. Filter by "App Engine"

### Cost
- **Free tier**: 28 instance hours/day
- After free tier: ~$0.05/hour for F1 instance
- Your app should stay in free tier with low traffic

---

## Alternative: Cloud Run (Containerized)

**Pros**: More flexible, better scaling  
**Cons**: Requires Docker knowledge

Stick with App Engine for simplicity.
