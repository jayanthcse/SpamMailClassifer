# 🟠 AWS Deployment Guide (Elastic Beanstalk)

## Prerequisites
- AWS account (free tier available)
- AWS CLI installed (optional but helpful)

## Method: AWS Elastic Beanstalk (Easiest AWS Option)

### Step 1: Prepare Application

Create a configuration file for Elastic Beanstalk:

Create `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app:app
```

### Step 2: Create ZIP File

```bash
# Include all files except .git
zip -r spam-classifier.zip . -x "*.git*" -x "*__pycache__*"
```

Or on Windows PowerShell:
```powershell
Compress-Archive -Path * -DestinationPath spam-classifier.zip -Force
```

### Step 3: Deploy via AWS Console

1. **Go to**: https://console.aws.amazon.com/elasticbeanstalk
2. **Click**: "Create Application"
3. **Fill in**:
   - Application name: `spam-classifier`
   - Platform: `Python`
   - Platform branch: `Python 3.11`
   - Platform version: (latest)
4. **Application code**: 
   - Select "Upload your code"
   - Click "Choose file"
   - Upload `spam-classifier.zip`
5. **Click**: "Create application"
6. **Wait**: 5-10 minutes for deployment

### Step 4: Access Your App

- URL will be: `spam-classifier.us-east-1.elasticbeanstalk.com`
- Click the URL in the Elastic Beanstalk dashboard

### Troubleshooting

**Check logs:**
1. Go to Elastic Beanstalk dashboard
2. Click "Logs" in left menu
3. Click "Request Logs" → "Last 100 Lines"

**Common issues:**
- If app doesn't start, check if `requirements.txt` is correct
- Model files might be too large for free tier (6MB should be fine)

### Cost
- **Free tier**: 750 hours/month for 12 months
- After free tier: ~$25/month for t2.micro instance

---

## Alternative: AWS Lambda + API Gateway (Serverless)

**Pros**: Pay per request, very cheap  
**Cons**: More complex setup, 50MB deployment limit (your models are 6MB, so OK)

This requires more AWS knowledge - stick with Elastic Beanstalk for simplicity.
