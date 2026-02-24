from src.app.api import app

# This ensures that 'app' is definitely available for Vercel to find
# even if there are weird namespace issues.
app = app
